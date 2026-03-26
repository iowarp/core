/*
 * CoroTidFix — LLVM pass plugin for GPU coroutine threadIdx.x preservation.
 *
 * Problem: clang-cuda's CoroSplit outlines coroutine resume paths into
 * separate functions. When one coroutine resumes another via
 * llvm.coro.resume (an indirect call through the frame), the resumed
 * function sees threadIdx.x == 0 for all warp lanes instead of the
 * correct per-lane hardware value.
 *
 * Fix: Before each llvm.coro.resume call, save %tid.x into the callee's
 * coroutine frame at a fixed negative offset (FrameHeader::lane_id_,
 * offset -4 from the frame pointer). In .resume functions, replace all
 * reads of the %tid.x special register with a load from that frame slot.
 *
 * This pass runs after CoroSplit (which creates .resume functions) and
 * only applies to the nvptx/nvptx64 targets.
 *
 * Load via: clang++ -fpass-plugin=<path>/libCoroTidFix.so
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

/// Fixed offset from the coroutine frame pointer where we store tid.x.
/// This corresponds to FrameHeader::lane_id_ which is at frame_ptr - 4.
/// FrameHeader is { u32 parallelism_; u32 lane_id_; } prepended before
/// the coroutine frame by operator new in gpu_coroutine.h.
static constexpr int64_t TID_FRAME_OFFSET = -4;

/// Check if a function is a coroutine resume function (created by CoroSplit).
/// CoroSplit names them <original>.resume and they take a single ptr arg.
static bool isResumeFunction(const Function &F) {
  if (!F.getName().ends_with(".resume"))
    return false;
  // Resume functions take exactly one argument: the frame pointer
  if (F.arg_size() != 1)
    return false;
  if (!F.getArg(0)->getType()->isPointerTy())
    return false;
  return true;
}

/// Get or declare the @llvm.nvvm.read.ptx.sreg.tid.x intrinsic.
static Function *getTidXIntrinsic(Module &M) {
  return Intrinsic::getDeclaration(&M, Intrinsic::nvvm_read_ptx_sreg_tid_x);
}

/// After CoroEarly, llvm.coro.resume is lowered to:
///   %fn_ptr = load ptr, ptr %frame   (resume fn is first field in frame)
///   call void %fn_ptr(ptr %frame)
/// We find indirect calls where the callee is loaded from the first
/// argument, and save tid.x into the frame before the call.
static bool instrumentCoroResumeCalls(Function &F, Function *TidXFn) {
  bool Changed = false;
  SmallVector<CallInst *, 4> ResumeCalls;

  for (auto &BB : F) {
    for (auto &I : BB) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI)
        continue;
      if (CI->arg_size() < 1)
        continue;

      Value *FrameArg = CI->getArgOperand(0);

      // Pattern 1: Indirect call — callee loaded from frame
      if (!CI->getCalledFunction()) {
        Value *Callee = CI->getCalledOperand();
        auto *Load = dyn_cast<LoadInst>(Callee);
        if (Load && Load->getPointerOperand() == FrameArg) {
          ResumeCalls.push_back(CI);
          continue;
        }
      }

      // Pattern 2: Direct call to a .resume function
      if (auto *Callee = CI->getCalledFunction()) {
        if (Callee->getName().ends_with(".resume")) {
          ResumeCalls.push_back(CI);
        }
      }
    }
  }

  for (auto *CI : ResumeCalls) {
    Value *FramePtr = CI->getArgOperand(0);
    IRBuilder<> Builder(CI);

    Value *TidX = Builder.CreateCall(TidXFn, {}, "tid.x.save");
    Value *Slot = Builder.CreateGEP(
        Builder.getInt8Ty(), FramePtr,
        Builder.getInt64(TID_FRAME_OFFSET), "tid.slot");
    Builder.CreateStore(TidX, Slot);

    Changed = true;
  }

  return Changed;
}

/// In .resume functions: replace all reads of %tid.x with a load from
/// the saved slot in the coroutine frame.
static bool fixResumeFunction(Function &F, Function *TidXFn) {
  // The frame pointer is the first (and only) argument
  Value *FramePtr = F.getArg(0);

  // Load the saved tid.x from the frame at entry
  IRBuilder<> EntryBuilder(&*F.getEntryBlock().getFirstInsertionPt());
  Value *Slot = EntryBuilder.CreateGEP(
      EntryBuilder.getInt8Ty(), FramePtr,
      EntryBuilder.getInt64(TID_FRAME_OFFSET), "tid.slot");
  Value *SavedTid = EntryBuilder.CreateLoad(
      EntryBuilder.getInt32Ty(), Slot, "tid.x.restored");

  // Find and replace all tid.x intrinsic calls
  SmallVector<CallInst *, 4> TidCalls;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        if (auto *Callee = CI->getCalledFunction()) {
          if (Callee->getIntrinsicID() == Intrinsic::nvvm_read_ptx_sreg_tid_x) {
            TidCalls.push_back(CI);
          }
        }
      }
    }
  }

  if (TidCalls.empty())
    return false;

  for (auto *CI : TidCalls) {
    CI->replaceAllUsesWith(SavedTid);
    CI->eraseFromParent();
  }

  return true;
}

struct CoroTidFixPass : PassInfoMixin<CoroTidFixPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    StringRef Triple = M.getTargetTriple();
    errs() << "[CoroTidFix] Module: " << M.getName()
           << " triple: " << Triple << "\n";

    if (!Triple.starts_with("nvptx")) {
      errs() << "[CoroTidFix] Skipping non-NVPTX module\n";
      return PreservedAnalyses::all();
    }

    Function *TidXFn = getTidXIntrinsic(M);
    if (!TidXFn) {
      errs() << "[CoroTidFix] No tid.x intrinsic found\n";
      return PreservedAnalyses::all();
    }

    bool Changed = false;
    unsigned ResumeCallsFixed = 0;
    unsigned ResumeFnsFixed = 0;

    for (auto &F : M) {
      if (F.isDeclaration())
        continue;
      if (F.getName().contains("Run") || F.getName().contains("resume") ||
          F.getName().contains("destroy")) {
        errs() << "[CoroTidFix] Function: " << F.getName() << "\n";
      }
      if (F.getName().contains("Run") || F.getName().ends_with(".resume")) {
        for (auto &BB : F) {
          for (auto &I : BB) {
            if (auto *CI = dyn_cast<CallInst>(&I)) {
              if (CI->getCalledFunction())
                errs() << "[CoroTidFix] " << F.getName() << " calls "
                       << CI->getCalledFunction()->getName() << "\n";
              else
                errs() << "[CoroTidFix] " << F.getName()
                       << " indirect call: " << *CI << "\n";
            }
          }
        }
      }
      if (instrumentCoroResumeCalls(F, TidXFn)) {
        Changed = true;
        ResumeCallsFixed++;
      }
    }

    for (auto &F : M) {
      if (F.isDeclaration())
        continue;
      if (F.getName().ends_with(".resume")) {
        errs() << "[CoroTidFix] Found .resume fn: " << F.getName()
               << " args=" << F.arg_size() << "\n";
        if (fixResumeFunction(F, TidXFn)) {
          Changed = true;
          ResumeFnsFixed++;
        }
      }
    }

    errs() << "[CoroTidFix] Fixed " << ResumeCallsFixed
           << " coro.resume callers, " << ResumeFnsFixed
           << " resume functions\n";

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

} // anonymous namespace

// Plugin registration
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CoroTidFix", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            // Register as a module pass running after the optimization pipeline
            // (after CoroSplit has created .resume functions)
            PB.registerOptimizerLastEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel) {
                  MPM.addPass(CoroTidFixPass());
                });
          }};
}
