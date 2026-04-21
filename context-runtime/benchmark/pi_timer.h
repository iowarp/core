#pragma once
#include <stdint.h>

namespace chi {

/**
 * Bypasses the OS kernel to read the raw ARM Cortex-A72 hardware tick counter.
 * Reads the CNTVCT_EL0 virtual count register directly from userspace.
 * Overhead: ~1-2 CPU cycles (single MRS instruction).
 *
 * Returns the virtual counter value in ticks (frequency depends on system).
 * On Raspberry Pi 4, typically 54 MHz (i.e., ~18.5 nanoseconds per tick).
 */
inline uint64_t get_cntvct_el0() {
  uint64_t tval;
  asm volatile("mrs %0, cntvct_el0" : "=r"(tval));
  return tval;
}

/**
 * Calibration loop to measure and prove timer overhead is functionally zero.
 * This should return 1-2 ticks (approximately 50-100 nanoseconds on Pi 4).
 * Use this to verify that your timing infrastructure isn't adding measurable
 * latency.
 */
inline uint64_t calibrate_timer_overhead() {
  uint64_t start = get_cntvct_el0();
  uint64_t end = get_cntvct_el0();
  return end - start;
}

}  // namespace chi
