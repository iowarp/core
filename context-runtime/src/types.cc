/**
 * Type implementations
 */

#include "chimaera/types.h"
#include <sstream>
#include <stdexcept>

namespace chi {

UniqueId UniqueId::FromString(const std::string& str) {
  // Parse format "major.minor"
  size_t dot_pos = str.find('.');
  if (dot_pos == std::string::npos) {
    throw std::invalid_argument("Invalid UniqueId format, expected 'major.minor'");
  }

  try {
    u32 major = std::stoul(str.substr(0, dot_pos));
    u32 minor = std::stoul(str.substr(dot_pos + 1));
    return UniqueId(major, minor);
  } catch (const std::exception& e) {
    throw std::invalid_argument("Failed to parse UniqueId: " + std::string(e.what()));
  }
}

}  // namespace chi
