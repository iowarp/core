/**
 * Acropolis set-depth CLI.
 *
 * Usage:
 *   acropolis-depth set <path> --level N
 *   acropolis-depth set <path> --level N --recursive
 *   acropolis-depth get <path>
 *   acropolis-depth clear <path>
 *
 * Stores the chosen depth level in the POSIX extended attribute
 *   user.acropolis.depth            (per-file)
 *   user.acropolis.depth_recursive  (per-directory, applies to children at ingest)
 *
 * The DepthController in the CTE runtime reads these xattrs on every PutBlob
 * to decide how deeply to index the file.
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
      "Levels:\n"
      "  0  Name only         (~0 cost)\n"
      "  1  Stat metadata     (~0 cost)\n"
      "  2  Format extraction (~ms)\n"
      "  3  Embedding         (~$0.001-0.01)\n"
      "  4  Deep content      (~$0.01-10+)\n";
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
      level = std::atoi(argv[++i]);
    } else if (a == "--recursive") {
      recursive = true;
    } else if (path.empty()) {
      path = a;
    } else {
      std::cerr << "unexpected argument: " << a << "\n";
      return 1;
    }
  }

  if (path.empty() || level < 0 || level > 4) {
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
  std::cout << "set " << attr << "=" << level << " on " << path << "\n";
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
  std::cout << "file:       " << (f.empty() ? "(unset)" : f) << "\n";
  std::cout << "directory:  " << (d.empty() ? "(unset)" : d) << "\n";
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
