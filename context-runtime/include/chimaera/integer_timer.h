// Copyright 2025 Chimaera Project
// Licensed under the Apache License, Version 2.0

#ifndef CHIMAERA_INCLUDE_CHIMAERA_INTEGER_TIMER_H_
#define CHIMAERA_INCLUDE_CHIMAERA_INTEGER_TIMER_H_

#include "chimaera/types.h"

namespace chi {

/**
 * Integer-based timepoint for performance optimization.
 * Uses a simple u64 counter instead of real time measurements.
 */
class IntegerTimepoint {
 public:
  u64 microseconds_;  /**< Time in microseconds since start */

  /** Default constructor */
  IntegerTimepoint() : microseconds_(0) {}

  /** Constructor with microseconds value */
  explicit IntegerTimepoint(u64 us) : microseconds_(us) {}

  /** Get microseconds value */
  u64 GetUsec() const { return microseconds_; }

  /** Comparison operators */
  bool operator<(const IntegerTimepoint &other) const {
    return microseconds_ < other.microseconds_;
  }

  bool operator>(const IntegerTimepoint &other) const {
    return microseconds_ > other.microseconds_;
  }

  bool operator<=(const IntegerTimepoint &other) const {
    return microseconds_ <= other.microseconds_;
  }

  bool operator>=(const IntegerTimepoint &other) const {
    return microseconds_ >= other.microseconds_;
  }

  bool operator==(const IntegerTimepoint &other) const {
    return microseconds_ == other.microseconds_;
  }

  bool operator!=(const IntegerTimepoint &other) const {
    return microseconds_ != other.microseconds_;
  }

  /** Addition operator */
  IntegerTimepoint operator+(const IntegerTimepoint &other) const {
    return IntegerTimepoint(microseconds_ + other.microseconds_);
  }

  /** Subtraction operator */
  IntegerTimepoint operator-(const IntegerTimepoint &other) const {
    return IntegerTimepoint(microseconds_ - other.microseconds_);
  }

  /** Addition assignment operator */
  IntegerTimepoint& operator+=(const IntegerTimepoint &other) {
    microseconds_ += other.microseconds_;
    return *this;
  }

  /** Subtraction assignment operator */
  IntegerTimepoint& operator-=(const IntegerTimepoint &other) {
    microseconds_ -= other.microseconds_;
    return *this;
  }
};

/**
 * Integer-based timer for performance optimization.
 * Uses a static counter instead of real time measurements.
 */
class IntegerTimer {
 public:
  static u64 current_time_us_;  /**< Current fake time in microseconds */

  /**
   * Get current timepoint.
   * @return Current IntegerTimepoint
   */
  static IntegerTimepoint Now() {
    return IntegerTimepoint(current_time_us_);
  }

  /**
   * Increment time by 1 microsecond.
   */
  static void Increment() {
    ++current_time_us_;
  }

  /**
   * Reset time to 0.
   */
  static void Reset() {
    current_time_us_ = 0;
  }
};

}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_INTEGER_TIMER_H_
