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

#include "adapter/cae_config.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>

#include "hermes_shm/util/config_parse.h"
#include "hermes_shm/util/logging.h"
#include "wrp_cte/core/content_transfer_engine.h"

namespace wrp::cae {

// Define global pointer variable in source file
HSHM_DEFINE_GLOBAL_PTR_VAR_CC(wrp::cae::CaeConfig, g_cae_config);

bool CaeConfig::LoadFromFile(const std::string& config_path) {
  if (config_path.empty()) {
    HLOG(kWarning, "Empty config path provided for CAE configuration");
    return false;
  }

  if (!std::filesystem::exists(config_path)) {
    HLOG(kWarning, "CAE config file does not exist: {}", config_path);
    return false;
  }

  try {
    YAML::Node config = YAML::LoadFile(config_path);
    return LoadFromYaml(config);
  } catch (const YAML::Exception& e) {
    HLOG(kError, "Failed to load CAE config from file {}: {}", config_path,
         e.what());
    return false;
  }
}

bool CaeConfig::LoadFromString(const std::string& yaml_content) {
  if (yaml_content.empty()) {
    HLOG(kWarning, "Empty YAML content provided for CAE configuration");
    return false;
  }

  try {
    YAML::Node config = YAML::Load(yaml_content);
    return LoadFromYaml(config);
  } catch (const YAML::Exception& e) {
    HLOG(kError, "Failed to load CAE config from YAML string: {}", e.what());
    return false;
  }
}

bool CaeConfig::LoadFromYaml(const YAML::Node& config) {
  try {
    patterns_.clear();

    // Load include patterns
    if (config["include"]) {
      const auto& include_node = config["include"];
      if (include_node.IsSequence()) {
        for (const auto& pattern_node : include_node) {
          if (pattern_node.IsScalar()) {
            std::string pattern = pattern_node.as<std::string>();
            if (!pattern.empty()) {
              // Expand environment variables in the pattern
              std::string expanded_pattern =
                  hshm::ConfigParse::ExpandPath(pattern);
              patterns_.emplace_back(expanded_pattern, true);  // true = include

              // Log if pattern was expanded
              if (expanded_pattern != pattern) {
                HLOG(kDebug, "Expanded include pattern: {} -> {}", pattern,
                     expanded_pattern);
              }
            }
          }
        }
      } else {
        HLOG(kError, "CAE config 'include' must be a sequence");
        return false;
      }
    }

    // Load exclude patterns
    if (config["exclude"]) {
      const auto& exclude_node = config["exclude"];
      if (exclude_node.IsSequence()) {
        for (const auto& pattern_node : exclude_node) {
          if (pattern_node.IsScalar()) {
            std::string pattern = pattern_node.as<std::string>();
            if (!pattern.empty()) {
              // Expand environment variables in the pattern
              std::string expanded_pattern =
                  hshm::ConfigParse::ExpandPath(pattern);
              patterns_.emplace_back(expanded_pattern,
                                     false);  // false = exclude

              // Log if pattern was expanded
              if (expanded_pattern != pattern) {
                HLOG(kDebug, "Expanded exclude pattern: {} -> {}", pattern,
                     expanded_pattern);
              }
            }
          }
        }
      } else {
        HLOG(kError, "CAE config 'exclude' must be a sequence");
        return false;
      }
    }

    // Sort patterns by length in descending order (most specific first)
    std::sort(patterns_.begin(), patterns_.end(),
              [](const PathPattern& a, const PathPattern& b) {
                return a.pattern.length() > b.pattern.length();
              });

    // Load adapter page size
    if (config["adapter_page_size"]) {
      adapter_page_size_ = config["adapter_page_size"].as<size_t>();
      if (adapter_page_size_ == 0) {
        HLOG(kWarning, "Invalid adapter page size 0, using default 4096");
        adapter_page_size_ = 4096;
      }
    }

    // Load interception enabled setting (optional, defaults to true)
    if (config["interception_enabled"]) {
      interception_enabled_ = config["interception_enabled"].as<bool>();
    }

    size_t include_count =
        std::count_if(patterns_.begin(), patterns_.end(),
                      [](const PathPattern& p) { return p.include; });
    size_t exclude_count = patterns_.size() - include_count;

    HLOG(kInfo,
         "CAE config loaded: {} include patterns, {} exclude patterns, "
         "page size {} bytes, interception {}",
         include_count, exclude_count, adapter_page_size_,
         interception_enabled_ ? "enabled" : "disabled");
    return true;

  } catch (const YAML::Exception& e) {
    HLOG(kError, "Failed to parse CAE config YAML: {}", e.what());
    return false;
  }
}

bool CaeConfig::SaveToFile(const std::string& config_path) const {
  if (config_path.empty()) {
    HLOG(kError, "Empty config path provided for saving CAE configuration");
    return false;
  }

  try {
    // Create parent directory if it doesn't exist
    std::filesystem::path file_path(config_path);
    if (file_path.has_parent_path()) {
      std::filesystem::create_directories(file_path.parent_path());
    }

    std::ofstream file(config_path);
    if (!file.is_open()) {
      HLOG(kError, "Failed to open CAE config file for writing: {}",
           config_path);
      return false;
    }

    file << ToYamlString();
    file.close();

    HLOG(kInfo, "CAE config saved to: {}", config_path);
    return true;

  } catch (const std::exception& e) {
    HLOG(kError, "Failed to save CAE config to file {}: {}", config_path,
         e.what());
    return false;
  }
}

std::string CaeConfig::ToYamlString() const {
  YAML::Node config;

  // Separate include and exclude patterns
  YAML::Node include_list;
  YAML::Node exclude_list;

  for (const auto& pattern : patterns_) {
    if (pattern.include) {
      include_list.push_back(pattern.pattern);
    } else {
      exclude_list.push_back(pattern.pattern);
    }
  }

  config["include"] = include_list;
  config["exclude"] = exclude_list;

  // Add adapter page size
  config["adapter_page_size"] = adapter_page_size_;

  // Add interception enabled setting
  config["interception_enabled"] = interception_enabled_;

  YAML::Emitter emitter;
  emitter << config;

  return emitter.c_str();
}

bool CaeConfig::IsPathTracked(const std::string& path) const {
  // Check global interception flag first
  if (!interception_enabled_) {
    return false;
  }

  // Check if CTE is not initialized yet
  auto* cte_manager = CTE_MANAGER;
  if (cte_manager != nullptr && !cte_manager->IsInitialized()) {
    return false;
  }

  // If no patterns configured, use smart defaults
  // Intercept user data paths but exclude system paths
  if (patterns_.empty()) {
    // Auto-exclude system paths to avoid interfering with MPI, libraries, etc.
    if (path.find("/lib") == 0 || path.find("/usr/lib") == 0 ||
        path.find("/lib64") == 0 || path.find("/usr/lib64") == 0 ||
        path.find("/bin") == 0 || path.find("/usr/bin") == 0 ||
        path.find("/sbin") == 0 || path.find("/usr/sbin") == 0 ||
        path.find("/etc") == 0 || path.find("/dev") == 0 ||
        path.find("/sys") == 0 || path.find("/proc") == 0 ||
        path.find("/run") == 0 || path.find("/boot") == 0) {
      return false;
    }

    // Include common data paths by default
    if (path.find("/tmp") == 0 || path.find("/scratch") == 0 ||
        path.find("/data") == 0 || path.find("/home") == 0 ||
        path.find("/mnt") == 0 || path.find("/var/tmp") == 0) {
      HLOG(kDebug, "CAE: Auto-including user data path: {}", path);
      return true;
    }

    // Default: exclude unknown paths
    return false;
  }

  // Check patterns in order of specificity (already sorted by length
  // descending)
  for (const auto& pattern_entry : patterns_) {
    try {
      std::regex pattern_regex(pattern_entry.pattern);
      if (std::regex_search(path, pattern_regex)) {
        // First match determines result
        return pattern_entry.include;
      }
    } catch (const std::regex_error& e) {
      HLOG(kWarning, "Invalid regex pattern '{}': {}", pattern_entry.pattern,
           e.what());
      continue;
    }
  }

  // No pattern matched - exclude by default
  return false;
}

void CaeConfig::AddIncludePattern(const std::string& pattern) {
  if (pattern.empty()) {
    return;
  }

  patterns_.emplace_back(pattern, true);

  // Re-sort by length (descending) to maintain specificity order
  std::sort(patterns_.begin(), patterns_.end(),
            [](const PathPattern& a, const PathPattern& b) {
              return a.pattern.length() > b.pattern.length();
            });

  HLOG(kDebug, "Added include pattern: {}", pattern);
}

void CaeConfig::AddExcludePattern(const std::string& pattern) {
  if (pattern.empty()) {
    return;
  }

  patterns_.emplace_back(pattern, false);

  // Re-sort by length (descending) to maintain specificity order
  std::sort(patterns_.begin(), patterns_.end(),
            [](const PathPattern& a, const PathPattern& b) {
              return a.pattern.length() > b.pattern.length();
            });

  HLOG(kDebug, "Added exclude pattern: {}", pattern);
}

void CaeConfig::ClearPatterns() {
  patterns_.clear();
  HLOG(kDebug, "Cleared all patterns");
}

bool WRP_CAE_CONFIG_INIT(const std::string& config_path) {
  // Check if CTE is still initializing
  auto* cte_manager = CTE_MANAGER;
  if (cte_manager != nullptr && !cte_manager->IsInitialized()) {
    return false;
  }

  auto* config = WRP_CAE_CONF;

  // Determine config path: use provided path, fallback to environment variable
  std::string actual_config_path = config_path;
  if (actual_config_path.empty()) {
    actual_config_path = hshm::SystemInfo::Getenv("WRP_CAE_CONF");
  }

  // Load from file if path is available
  if (!actual_config_path.empty()) {
    config->LoadFromFile(actual_config_path);
  }

  return true;
}

}  // namespace wrp::cae