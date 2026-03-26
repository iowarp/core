/*
 * CoroTidFix — LLVM pass plugin for GPU coroutine threadIdx.x preservation.
 *
 * Problem: clang-cuda's CoroSplit outlines coroutine resume paths into
 * separate functions where threadIdx.x reads the wrong value. Inner
 * coroutines created from within a .resume function also get wrong values.
 *
 * Fix: Use a per-lane shared memory slot to pass the correct tid.x.
 * - In .resume functions: load tid.x from FrameHeader, store to shared mem
 * - In all non-kernel device functions: read tid.x from shared mem instead
 *   of the hardware register
 * - In kernel functions: store hardware tid.x to shared mem at entry
 *
 * Shared memory layout: __shared__ u32 _coro_tid_x[32];
 * Indexed by hardware %laneid (always correct).
 *
 * Only applies to nvptx/nvptx64 targets.
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
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

static constexpr int64_t TID_FRAME_OFFSET = -4;
static constexpr unsigned SHARED_ADDR_SPACE = 3;  // NVPTX shared memory

static Function *getTidXIntrinsic(Module &M) {
  return Intrinsic::getDeclaration(&M, Intrinsic::nvvm_read_ptx_sreg_tid_x);
}

static Function *getLaneIdIntrinsic(Module &M) {
  return Intrinsic::getDeclaration(&M, Intrinsic::nvvm_read_ptx_sreg_laneid);
}

/// Get or create the shared memory array: __shared__ u32 _coro_tid_x[32]
static GlobalVariable *getOrCreateSharedTidArray(Module &M) {
  auto *Existing = M.getGlobalVariable("_coro_tid_x", true);
  if (Existing) return Existing;

  auto *ArrTy = ArrayType::get(Type::getInt32Ty(M.getContext()), 32);
  auto *GV = new GlobalVariable(
      M, ArrTy, false, GlobalValue::InternalLinkage,
      UndefValue::get(ArrTy), "_coro_tid_x",
      nullptr, GlobalValue::NotThreadLocal, SHARED_ADDR_SPACE);
  GV->setAlignment(Align(128));
  return GV;
}

static void collectTidXCalls(Function &F, SmallVectorImpl<CallInst *> &Calls) {
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        if (auto *Callee = CI->getCalledFunction()) {
          if (Callee->getIntrinsicID() == Intrinsic::nvvm_read_ptx_sreg_tid_x) {
            Calls.push_back(CI);
          }
        }
      }
    }
  }
}

static bool isKernelFunction(const Function &F, const Module &M) {
  auto *NvvmAnnot = M.getNamedMetadata("nvvm.annotations");
  if (!NvvmAnnot) return false;
  for (auto *Op : NvvmAnnot->operands()) {
    if (Op->getNumOperands() >= 3) {
      if (auto *FnMD = dyn_cast<ValueAsMetadata>(Op->getOperand(0))) {
        if (FnMD->getValue() == &F) {
          if (auto *KindMD = dyn_cast<MDString>(Op->getOperand(1))) {
            if (KindMD->getString() == "kernel") return true;
          }
        }
      }
    }
  }
  return false;
}

/// Load tid.x from shared memory: _coro_tid_x[%laneid]
static Value *loadTidFromShared(IRBuilder<> &Builder, GlobalVariable *SharedArr,
                                 Function *LaneIdFn) {
  Value *LaneId = Builder.CreateCall(LaneIdFn, {}, "laneid");
  Value *Indices[] = {Builder.getInt32(0), LaneId};
  Value *Ptr = Builder.CreateInBoundsGEP(SharedArr->getValueType(),
                                          SharedArr, Indices, "tid.shared.ptr");
  return Builder.CreateLoad(Builder.getInt32Ty(), Ptr, "tid.x.shared");
}

/// Store tid.x to shared memory: _coro_tid_x[%laneid] = val
static void storeTidToShared(IRBuilder<> &Builder, Value *TidVal,
                              GlobalVariable *SharedArr, Function *LaneIdFn) {
  Value *LaneId = Builder.CreateCall(LaneIdFn, {}, "laneid");
  Value *Indices[] = {Builder.getInt32(0), LaneId};
  Value *Ptr = Builder.CreateInBoundsGEP(SharedArr->getValueType(),
                                          SharedArr, Indices, "tid.shared.ptr");
  Builder.CreateStore(TidVal, Ptr);
}

/// Kernel entry: store hardware tid.x to shared memory.
static bool instrumentKernel(Function &F, Function *TidXFn,
                              Function *LaneIdFn, GlobalVariable *SharedArr) {
  SmallVector<CallInst *, 4> TidCalls;
  collectTidXCalls(F, TidCalls);
  if (TidCalls.empty()) return false;

  // Store tid.x to shared at kernel entry
  IRBuilder<> Builder(&*F.getEntryBlock().getFirstInsertionPt());
  Value *TidX = Builder.CreateCall(TidXFn, {}, "tid.x.hw");
  storeTidToShared(Builder, TidX, SharedArr, LaneIdFn);

  return false;  // Don't replace reads in kernels — they're correct
}

/// .resume function: load tid.x from FrameHeader, store to shared,
/// then replace all tid.x reads.
static bool fixResumeFunction(Function &F, Function *LaneIdFn,
                               GlobalVariable *SharedArr) {
  SmallVector<CallInst *, 4> TidCalls;
  collectTidXCalls(F, TidCalls);

  Value *FramePtr = F.getArg(0);
  IRBuilder<> Builder(&*F.getEntryBlock().getFirstInsertionPt());

  // Load from FrameHeader
  Value *Slot = Builder.CreateGEP(
      Builder.getInt8Ty(), FramePtr,
      Builder.getInt64(TID_FRAME_OFFSET), "tid.slot");
  Value *SavedTid = Builder.CreateLoad(
      Builder.getInt32Ty(), Slot, "tid.x.frame");

  // Store to shared memory so callees can find it
  storeTidToShared(Builder, SavedTid, SharedArr, LaneIdFn);

  // Replace tid.x reads
  if (!TidCalls.empty()) {
    for (auto *CI : TidCalls) {
      CI->replaceAllUsesWith(SavedTid);
      CI->eraseFromParent();
    }
  }
  return true;
}

/// Non-kernel, non-resume device function: replace tid.x reads with
/// loads from shared memory.
static bool fixDeviceFunction(Function &F, Function *LaneIdFn,
                               GlobalVariable *SharedArr) {
  SmallVector<CallInst *, 4> TidCalls;
  collectTidXCalls(F, TidCalls);
  if (TidCalls.empty()) return false;

  // Load once from shared at function entry
  IRBuilder<> Builder(&*F.getEntryBlock().getFirstInsertionPt());
  Value *TidFromShared = loadTidFromShared(Builder, SharedArr, LaneIdFn);

  for (auto *CI : TidCalls) {
    CI->replaceAllUsesWith(TidFromShared);
    CI->eraseFromParent();
  }
  return true;
}

struct CoroTidFixPass : PassInfoMixin<CoroTidFixPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    StringRef Triple = M.getTargetTriple();
    if (!Triple.starts_with("nvptx"))
      return PreservedAnalyses::all();

    Function *TidXFn = getTidXIntrinsic(M);
    Function *LaneIdFn = getLaneIdIntrinsic(M);
    if (!TidXFn || !LaneIdFn)
      return PreservedAnalyses::all();

    GlobalVariable *SharedArr = getOrCreateSharedTidArray(M);
    bool Changed = false;

    // Process kernels first: store tid.x to shared at entry
    for (auto &F : M) {
      if (F.isDeclaration()) continue;
      if (isKernelFunction(F, M)) {
        Changed |= instrumentKernel(F, TidXFn, LaneIdFn, SharedArr);
      }
    }

    // Process .resume functions: load from FrameHeader, store to shared
    for (auto &F : M) {
      if (F.isDeclaration()) continue;
      if (F.getName().ends_with(".resume") && F.arg_size() == 1 &&
          F.getArg(0)->getType()->isPointerTy()) {
        Changed |= fixResumeFunction(F, LaneIdFn, SharedArr);
      }
    }

    // Process all other device functions: read from shared
    for (auto &F : M) {
      if (F.isDeclaration()) continue;
      if (F.getName().ends_with(".resume") || isKernelFunction(F, M)) continue;
      Changed |= fixDeviceFunction(F, LaneIdFn, SharedArr);
    }

    errs() << "[CoroTidFix] " << M.getName() << ": done\n";
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

} // anonymous namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CoroTidFix", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerOptimizerLastEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel) {
                  MPM.addPass(CoroTidFixPass());
                });
          }};
}
