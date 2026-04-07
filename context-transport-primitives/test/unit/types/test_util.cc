/*
 * Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology
 * All rights reserved.
 *
 * This file is part of IOWarp Core.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#define HSHM_COMPILING_DLL

#include "basic_test.h"
#include "hermes_shm/data_structures/internal/shm_archive.h"
#include "hermes_shm/thread/thread_model_manager.h"
#include "hermes_shm/util/auto_trace.h"
#include "hermes_shm/util/config_parse.h"
#include "hermes_shm/util/formatter.h"
#include "hermes_shm/util/logging.h"
#include "hermes_shm/util/singleton.h"
#include "hermes_shm/util/type_switch.h"

TEST_CASE("TypeSwitch") {
  typedef hshm::type_switch<int, int, std::string, std::string, size_t,
                            size_t>::type internal_t;
  REQUIRE(std::is_same_v<internal_t, int>);

  typedef hshm::type_switch<size_t, int, std::string, std::string, size_t,
                            size_t>::type internal2_t;
  REQUIRE(std::is_same_v<internal2_t, size_t>);

  typedef hshm::type_switch<std::string, int, std::string, std::string, size_t,
                            size_t>::type internal3_t;
  REQUIRE(std::is_same_v<internal3_t, std::string>);

  typedef hshm::type_switch<std::vector<int>, int, std::string, std::string,
                            size_t, size_t>::type internal4_t;
  REQUIRE(std::is_same_v<internal4_t, int>);
}

TEST_CASE("TestPathParser") {
  hshm::SystemInfo::Setenv("PATH_PARSER_TEST", "HOME", true);
  auto x = hshm::ConfigParse::ExpandPath("${PATH_PARSER_TEST}/hello");
  hshm::SystemInfo::Unsetenv("PATH_PARSER_TEST");
  auto y = hshm::ConfigParse::ExpandPath("${PATH_PARSER_TEST}/hello");
  auto z = hshm::ConfigParse::ExpandPath("${HOME}/hello");
  REQUIRE(x == "HOME/hello");
  REQUIRE(y == "/hello");
  REQUIRE(z != "${HOME}/hello");
}

TEST_CASE("TestNumberParser") {
  REQUIRE(hshm::Unit<hshm::u64>::Kilobytes(1.5) == 1536);
  REQUIRE(hshm::Unit<hshm::u64>::Megabytes(1.5) == 1572864);
  REQUIRE(hshm::Unit<hshm::u64>::Gigabytes(1.5) == 1610612736);
  REQUIRE(hshm::Unit<hshm::u64>::Terabytes(1.5) == 1649267441664);
  REQUIRE(hshm::Unit<hshm::u64>::Petabytes(1.5) == 1688849860263936);

  std::pair<std::string, hshm::u64> sizes[] = {
      {"1", 1},
      {"1.5", 1},
      {"1KB", hshm::Unit<hshm::u64>::Kilobytes(1)},
      {"1.5MB", hshm::Unit<hshm::u64>::Megabytes(1.5)},
      {"1.5GB", hshm::Unit<hshm::u64>::Gigabytes(1.5)},
      {"2TB", hshm::Unit<hshm::u64>::Terabytes(2)},
      {"1.5PB", hshm::Unit<hshm::u64>::Petabytes(1.5)},
  };

  for (auto &[text, val] : sizes) {
    REQUIRE(hshm::ConfigParse::ParseSize(text) == val);
  }
  REQUIRE(hshm::ConfigParse::ParseSize("inf"));
}

TEST_CASE("TestSystemInfo") {
  // GetModuleDirectory: should return a non-empty path (exercises PATH_MAX buffer)
  std::string mod_dir = hshm::SystemInfo::GetModuleDirectory();
  REQUIRE(!mod_dir.empty());

  // GetMemfdDir / GetMemfdPath: exercises the /tmp/chimaera_<user> path helpers
  std::string memfd_dir = hshm::SystemInfo::GetMemfdDir();
  REQUIRE(!memfd_dir.empty());
  REQUIRE(memfd_dir.find("/tmp/") != std::string::npos);

  std::string memfd_path = hshm::SystemInfo::GetMemfdPath("test_shm");
  REQUIRE(!memfd_path.empty());
  REQUIRE(memfd_path.find("test_shm") != std::string::npos);

  // GetSharedLibExtension / GetLibrarySearchPathVar / GetPathListSeparator
  std::string ext = hshm::SystemInfo::GetSharedLibExtension();
  REQUIRE(!ext.empty());
  REQUIRE(ext == ".so" || ext == ".dylib" || ext == ".dll");

  std::string lib_var = hshm::SystemInfo::GetLibrarySearchPathVar();
  REQUIRE(!lib_var.empty());

  char sep = hshm::SystemInfo::GetPathListSeparator();
  REQUIRE((sep == ':' || sep == ';'));
}

TEST_CASE("TestTerminal") {
  std::cout << "\033[1m" << "Bold text" << "\033[0m" << std::endl;
  std::cout << "\033[4m" << "Underlined text" << "\033[0m" << std::endl;
  std::cout << "\033[31m" << "Red text" << "\033[0m" << std::endl;
  std::cout << "\033[32m" << "Green text" << "\033[0m" << std::endl;
  std::cout << "\033[33m" << "Yellow text" << "\033[0m" << std::endl;
  std::cout << "\033[34m" << "Blue text" << "\033[0m" << std::endl;
  std::cout << "\033[35m" << "Magenta text" << "\033[0m" << std::endl;
  std::cout << "\033[36m" << "Cyan text" << "\033[0m" << std::endl;
}

TEST_CASE("TestAutoTrace") {
  AUTO_TRACE(0);

  TIMER_START("Example");
  HSHM_THREAD_MODEL->SleepForUs(1000);
  TIMER_END();
}

TEST_CASE("TestLogger") {
  HLOG(kInfo, "I'm more likely to be printed: {}", 0);
  HLOG(kDebug, "I'm not likely to be printed: {}", 10);

  HSHM_LOG->DisableCode(kDebug);
  HLOG(kInfo, "I'm more likely to be printed (2): {}", 0);
  HLOG(kDebug, "I won't be printed: {}", 10);

  HLOG(kWarning, "I am a WARNING! Will NOT cause an EXIT!");
  HLOG(kError, "I am an ERROR! I will NOT cause an EXIT!");
}

TEST_CASE("TestFatalLogger", "[error=FatalError]") {
  HLOG(kFatal, "I will cause an EXIT!");
}

TEST_CASE("TestFormatter") {
  int rank = 0;
  int i = 0;

  PAGE_DIVIDE("Test with equivalent parameters") {
    std::string name = hshm::Formatter::format("bucket{}_{}", rank, i);
    REQUIRE(name == "bucket0_0");
  }

  PAGE_DIVIDE("Test with equivalent parameters at start") {
    std::string name = hshm::Formatter::format("{}bucket{}", rank, i);
    REQUIRE(name == "0bucket0");
  }

  PAGE_DIVIDE("Test with too many parameters") {
    std::string name = hshm::Formatter::format("bucket", rank, i);
    REQUIRE(name == "bucket");
  }

  PAGE_DIVIDE("Test with fewer parameters") {
    std::string name = hshm::Formatter::format("bucket{}_{}", rank);
    REQUIRE(name == "bucket{}_{}");
  }

  PAGE_DIVIDE("Test with parameters next to each other") {
    std::string name = hshm::Formatter::format("bucket{}{}", rank, i);
    REQUIRE(name == "bucket00");
  }
}
