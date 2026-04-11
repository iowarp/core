# aya-ebpf 0.1.1 API Migration Guide

## Summary

Successfully migrated the eBPF kernel program from the broken `ctx.arg()` API to the correct aya-ebpf 0.1.1 tracepoint argument access pattern.

## Changes Made

### 1. Import EbpfContext Trait

**Added import:**
```rust
use aya_ebpf::{
    macros::{map, tracepoint},
    maps::RingBuf,
    programs::TracePointContext,
    EbpfContext,  // NEW: Import EbpfContext trait for as_ptr() method
};
```

### 2. Defined Tracepoint Argument Structures

Instead of using `ctx.arg(N)` (which doesn't exist in aya-ebpf 0.1.1), we define proper C structures that match the kernel tracepoint format:

```rust
#[repr(C)]
struct SysEnterOpenatArgs {
    dfd: i32,
    filename: *const u8,
    flags: i32,
    mode: u64,
}

#[repr(C)]
struct SysExitOpenatArgs {
    ret: i64,
}

// Similar structures for read, write, close syscalls...
```

### 3. Access Arguments via Structure Cast

**Old (broken) pattern:**
```rust
unsafe fn get_openat_filename(ctx: &TracePointContext) -> *const u8 {
    ctx.arg(1) as *const u8  // ERROR: no method `arg`
}
```

**New (correct) pattern:**
```rust
#[tracepoint]
pub fn sys_enter_openat(ctx: TracePointContext) {
    let args = unsafe { &*(ctx.as_ptr() as *const SysEnterOpenatArgs) };
    let filename_ptr = args.filename;
    // ...
}
```

### 4. Fixed emit_event Signature

**Before:**
```rust
unsafe fn emit_event(event: &IoEvent) {
    if let Some(mut record) = EVENTS.reserve::<IoEvent>(0) {
        record.write(event);  // write() expects value, not reference
        record.submit(0);
    }
}

// Usage:
emit_event(&event);
```

**After:**
```rust
unsafe fn emit_event(event: IoEvent) {
    if let Some(mut record) = EVENTS.reserve::<IoEvent>(0) {
        record.write(event);  // Now receives value
        record.submit(0);
    }
}

// Usage:
emit_event(event);
```

### 5. Removed Unnecessary Unsafe

**Before:**
```rust
let pid_tgid = unsafe { aya_ebpf::helpers::bpf_get_current_pid_tgid() };
```

**After:**
```rust
let pid_tgid = aya_ebpf::helpers::bpf_get_current_pid_tgid();
```

The BPF helper function is already marked unsafe inside the function, so wrapping it in `unsafe` block is unnecessary.

## Key Concepts

### Tracepoint Argument Access in aya-ebpf 0.1.1

1. **No `.arg()` method**: The `TracePointContext` does not have an `.arg()` method in aya-ebpf 0.1.1
2. **Use `as_ptr()`**: Access the raw context pointer using `ctx.as_ptr()` (requires `EbpfContext` trait)
3. **Structure cast**: Cast the pointer to the appropriate tracepoint argument structure
4. **Match kernel format**: Structure fields must match the kernel's tracepoint format exactly

### BPF Helper Functions

BPF helper functions like `bpf_get_current_pid_tgid()` are called directly (no unsafe wrapper needed):
```rust
let pid_tgid = aya_ebpf::helpers::bpf_get_current_pid_tgid();
```

### Ring Buffer Submission

The `RingBuf::reserve()` pattern requires passing values (not references) to `record.write()`:
```rust
if let Some(mut record) = EVENTS.reserve::<IoEvent>(0) {
    record.write(event);  // event: IoEvent (value)
    record.submit(0);
}
```

## Verification

The eBPF program now compiles successfully:
```bash
cd interceptor-ebpf
cargo +nightly build --release
```

Output:
```
Finished `release` profile [optimized] target(s) in 0.13s
```

The generated binary is a valid eBPF ELF:
```bash
file interceptor-ebpf/target/bpfel-unknown-none/release/interceptor-ebpf
# Output: ELF 64-bit LSB relocatable, eBPF, version 1 (SYSV), not stripped
```

## Tracepoint Format Reference

For accurate structure definitions, check the kernel tracepoint format files:
```bash
cat /sys/kernel/debug/tracing/events/syscalls/sys_enter_openat/format
cat /sys/kernel/debug/tracing/events/syscalls/sys_exit_openat/format
```

Each tracepoint format shows the exact field order and types needed for the structure cast.

## Related Files

- `interceptor-ebpf/src/main.rs` - eBPF kernel program (migrated)
- `interceptor-ebpf-common/src/lib.rs` - Shared data structures (unchanged)
- `.cargo/config.toml` - Build configuration for bpfel-unknown-none target

## Build Requirements

- Rust nightly toolchain (for build-std support)
- Target: `bpfel-unknown-none`
- Profile: release (unstable features in .cargo/config.toml)

## Next Steps

1. Test the eBPF program with the user-space controller
2. Verify tracepoint attachment and event capture
3. Add additional syscall support as needed