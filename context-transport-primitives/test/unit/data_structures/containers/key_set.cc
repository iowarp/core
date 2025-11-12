//
// Created by llogan on 11/29/24.
//

#include "basic_test.h"
#include "test_init.h"
#include "list.h"

#include "hermes_shm/data_structures/all.h"

TEST_CASE("SpscKeySet") {
  hshm::spsc_key_set<size_t> count;
  count.Init(32);
  std::vector<hshm::size_t> keys(64);
  for (size_t i = 0; i < 64; ++i) {
    size_t entry = i;
    count.emplace(keys[i], entry);
  }

  for (size_t i = 0; i < 64; ++i) {
    size_t entry;
    count.pop(keys[i], entry);
    REQUIRE(entry == i);
  }
}

TEST_CASE("MpmcKeySet") {
  hshm::spsc_key_set<size_t> count;
  count.Init(32);
  std::vector<hshm::size_t> keys(32);

  for (size_t i = 0; i < 32; ++i) {
    size_t entry = i;
    count.emplace(keys[i], entry);
  }

  for (size_t i = 0; i < 32; ++i) {
    size_t entry;
    count.pop(keys[i], entry);
    REQUIRE(entry == i);
  }
}
