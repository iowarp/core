/**
 * Acropolis set-depth CLI.
 *
 * Usage:
 *   acropolis-depth set <path> --level N [--recursive]
 *   acropolis-depth set <path> --level NAME [--recursive]
 *   acropolis-depth get <path>
 *   acropolis-depth clear <path>
 *
 * Levels (numeric or named):
 *   0 | name      filename only
 *   1 | metadata  + format-specific extraction + embedding
 *   2 | content   L1 + caller-supplied LLM summary (driven by CAE)
 *
 * Stores the chosen depth in the POSIX extended attribute
 *   user.acropolis.depth            (per-file)
 *   user.acropolis.depth_recursive  (per-directory; applies to children at ingest)
 */

#include <sys/xattr.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {

constexpr const char *kAttrFile      = "user.acropolis.depth";
constexpr const char *kAttrRecursive = "user.acropolis.depth_recursive";

void PrintUsage() {
  std::cerr <<
      "Usage:\n"
      "  acropolis-depth set <path> --level N [--recursive]\n"
      "  acropolis-depth get <path>\n"
      "  acropolis-depth clear <path>\n"
      "\n"
      "Levels (numeric or named):\n"
      "  0 | name      filename only                 (~zero cost)\n"
      "  1 | metadata  + format extraction + embed   (~ms)\n"
      "  2 | content   L1 + LLM summary via CAE      (~LLM call)\n";
}

/** Accept "0".."2" or "name", "metadata", "content". Returns -1 if invalid. */
int ParseLevel(const std::string &s) {
  if (s == "0" || s == "name")     return 0;
  if (s == "1" || s == "metadata") return 1;
  if (s == "2" || s == "content")  return 2;
  // Tolerate raw integers within range
  char *end = nullptr;
  long v = std::strtol(s.c_str(), &end, 10);
  if (end != s.c_str() && *end == '\0' && v >= 0 && v <= 2) return static_cast<int>(v);
  return -1;
}

const char *LevelLabel(int level) {
  switch (level) {
    case 0: return "0 (name)";
    case 1: return "1 (metadata)";
    case 2: return "2 (content)";
  }
  return "unknown";
}

int CmdSet(int argc, char **argv) {
  if (argc < 5) { PrintUsage(); return 1; }
  std::string path;
  int level = -1;
  bool recursive = false;

  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--level") {
      if (i + 1 >= argc) { PrintUsage(); return 1; }
      level = ParseLevel(argv[++i]);
    } else if (a == "--recursive") {
      recursive = true;
    } else if (path.empty()) {
      path = a;
    } else {
      std::cerr << "unexpected argument: " << a << "\n";
      return 1;
    }
  }

  if (path.empty() || level < 0) {
    PrintUsage();
    return 1;
  }

  std::string val = std::to_string(level);
  const char *attr = recursive ? kAttrRecursive : kAttrFile;
  int rc = ::setxattr(path.c_str(), attr, val.data(), val.size(), 0);
  if (rc != 0) {
    std::perror("setxattr");
    return 2;
  }
  std::cout << "set " << attr << "=" << LevelLabel(level)
            << " on " << path << "\n";
  return 0;
}

int CmdGet(int argc, char **argv) {
  if (argc < 3) { PrintUsage(); return 1; }
  std::string path = argv[2];

  auto read_attr = [&](const char *attr) -> std::string {
    char buf[16] = {0};
    ssize_t n = ::getxattr(path.c_str(), attr, buf, sizeof(buf) - 1);
    if (n <= 0) return {};
    return std::string(buf, buf + n);
  };

  std::string f = read_attr(kAttrFile);
  std::string d = read_attr(kAttrRecursive);

  auto pretty = [](const std::string &s) -> std::string {
    if (s.empty()) return "(unset)";
    int v = std::atoi(s.c_str());
    return std::string(LevelLabel(v));
  };

  std::cout << "file:       " << pretty(f) << "\n";
  std::cout << "directory:  " << pretty(d) << "\n";
  return 0;
}

int CmdClear(int argc, char **argv) {
  if (argc < 3) { PrintUsage(); return 1; }
  std::string path = argv[2];
  ::removexattr(path.c_str(), kAttrFile);
  ::removexattr(path.c_str(), kAttrRecursive);
  std::cout << "cleared acropolis xattrs on " << path << "\n";
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) { PrintUsage(); return 1; }
  std::string cmd = argv[1];
  if (cmd == "set")   return CmdSet(argc, argv);
  if (cmd == "get")   return CmdGet(argc, argv);
  if (cmd == "clear") return CmdClear(argc, argv);
  PrintUsage();
  return 1;
}
