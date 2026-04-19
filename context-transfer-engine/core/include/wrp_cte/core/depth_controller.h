#ifndef WRPCTE_DEPTH_CONTROLLER_H_
#define WRPCTE_DEPTH_CONTROLLER_H_

/**
 * DepthController — Acropolis adaptive indexing depth.
 *
 * Runs 0..N level executors additively at ingest time to produce a single
 * payload (text + optional embedding) that is handed to the configured
 * KGBackend. The default depth is L0 (name only, ~zero cost); users opt into
 * deeper indexing per file / directory / format.
 *
 * Policy resolution order (highest priority first):
 *   1. Explicit target parameter passed to Index()
 *   2. File-level xattr:        user.acropolis.depth
 *   3. Directory-level xattr:   user.acropolis.depth_recursive
 *   4. Format default from YAML config
 *   5. Global default from YAML config (or L0 if no config)
 */

#include <wrp_cte/core/core_tasks.h>
#include <wrp_cte/core/embedding_client.h>

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __linux__
#include <sys/xattr.h>
#endif

namespace wrp_cte::core {

/** What the controller feeds to the configured KGBackend. */
struct IndexPayload {
  TagId tag_id;
  std::string text;                     ///< accumulated summary text
  std::vector<float> embedding;         ///< populated only at L3+
  IndexDepth depth_achieved = IndexDepth::kNameOnly;
};

/** YAML-driven default levels per format extension. */
struct DepthDefaults {
  IndexDepth global_default = IndexDepth::kNameOnly;
  std::map<std::string, IndexDepth> per_format;  ///< key: lowercase extension ("hdf5", "h5", "nc", ...)
};

/**
 * L2Extractor — pluggable per-format metadata extractor.
 *
 * Takes a file path, returns a human-readable summary string that describes
 * the file's internal structure (HDF5 dataset tree, Parquet schema, NetCDF
 * variables, etc.). The returned string is appended to the depth payload
 * and indexed by the configured backend.
 *
 * Implementations must be safe to call concurrently from multiple runtime
 * workers. They should degrade gracefully (return empty string) if the file
 * cannot be opened.
 */
using L2Extractor = std::function<std::string(const std::string &path)>;

class DepthController {
 public:
  DepthController() = default;

  /** Install a default-policy table loaded from YAML / config. */
  void SetDefaults(DepthDefaults defaults) {
    std::lock_guard<std::mutex> lk(mu_);
    defaults_ = std::move(defaults);
  }

  /** Configure the embedding client used for L3. */
  void SetEmbedder(EmbeddingClient embedder) {
    std::lock_guard<std::mutex> lk(mu_);
    embedder_ = std::move(embedder);
  }

  /**
   * Register an L2 extractor for a specific file extension. The extension
   * key is lowercase, without leading dot (e.g. "h5", "hdf5", "parquet").
   *
   * When L2 runs on a file with a matching extension, the registered
   * extractor is invoked and its output is appended to the payload text
   * (replacing the default content_kind=... tag).
   *
   * Later registrations for the same extension overwrite earlier ones.
   */
  void RegisterL2Extractor(const std::string &extension, L2Extractor fn) {
    std::lock_guard<std::mutex> lk(mu_);
    l2_extractors_[extension] = std::move(fn);
  }

  /** Remove any registered extractor for a given extension. */
  void UnregisterL2Extractor(const std::string &extension) {
    std::lock_guard<std::mutex> lk(mu_);
    l2_extractors_.erase(extension);
  }

  /** Query if an L2 extractor exists for an extension (mainly for tests). */
  bool HasL2Extractor(const std::string &extension) const {
    std::lock_guard<std::mutex> lk(mu_);
    return l2_extractors_.count(extension) > 0;
  }

  /** Resolve the effective target depth for a file/tag. */
  IndexDepth ResolvePolicy(const std::string &tag_name_or_path,
                           std::optional<IndexDepth> explicit_target) const {
    if (explicit_target.has_value()) return *explicit_target;

#ifdef __linux__
    // File-level xattr
    if (auto d = ReadDepthXattr(tag_name_or_path, "user.acropolis.depth"))
      return *d;
    // Directory-level xattr (walk up one parent)
    auto slash = tag_name_or_path.find_last_of('/');
    if (slash != std::string::npos) {
      std::string dir = tag_name_or_path.substr(0, slash);
      if (auto d = ReadDepthXattr(dir, "user.acropolis.depth_recursive"))
        return *d;
    }
#endif

    // Format-based default
    std::string ext = FileExtension(tag_name_or_path);
    std::lock_guard<std::mutex> lk(mu_);
    auto it = defaults_.per_format.find(ext);
    if (it != defaults_.per_format.end()) return it->second;
    return defaults_.global_default;
  }

  /**
   * Produce an IndexPayload for the given tag at the target depth.
   * Each level is additive — running level N executes levels 0..N.
   *
   * `tag_name_or_path` is the blob/file identifier (used for L0/L1).
   * `data` is the raw blob contents (used for L2/L4); may be empty.
   * `file_size` is the authoritative size (if `data` is truncated).
   */
  IndexPayload Index(const TagId &tag_id,
                     const std::string &tag_name_or_path,
                     const std::vector<char> &data,
                     uint64_t file_size,
                     IndexDepth target) const {
    IndexPayload out;
    out.tag_id = tag_id;

    std::string ext = FileExtension(tag_name_or_path);

    // Levels are additive — accumulate text into `out.text`.
    RunL0(out, tag_name_or_path, file_size);
    out.depth_achieved = IndexDepth::kNameOnly;
    if (target == IndexDepth::kNameOnly) return out;

    RunL1(out, tag_name_or_path, ext, data);
    out.depth_achieved = IndexDepth::kStatMeta;
    if (target == IndexDepth::kStatMeta) return out;

    RunL2Dispatch(out, tag_name_or_path, ext, data);
    out.depth_achieved = IndexDepth::kFormatExtract;
    if (target == IndexDepth::kFormatExtract) return out;

    RunL3(out);
    out.depth_achieved = IndexDepth::kEmbedding;
    if (target == IndexDepth::kEmbedding) return out;

    RunL4(out, ext, data);
    out.depth_achieved = IndexDepth::kDeepContent;
    return out;
  }

 private:
  // ---------- Level executors (implemented inline, header-only) ----------

  static void RunL0(IndexPayload &out, const std::string &name,
                    uint64_t file_size) {
    // Filename, path components, size
    out.text += "path=";
    out.text += name;
    out.text += " size=";
    out.text += std::to_string(file_size);
    std::string ext = FileExtension(name);
    if (!ext.empty()) {
      out.text += " ext=";
      out.text += ext;
    }
  }

  static void RunL1(IndexPayload &out, const std::string &name,
                    const std::string &ext,
                    const std::vector<char> &data) {
    (void)name;
    // Format sniffing from magic bytes (cheap, no external deps)
    std::string fmt = SniffFormat(ext, data);
    if (!fmt.empty()) {
      out.text += " format=";
      out.text += fmt;
    }
  }

  /**
   * L2 dispatch — looks up a registered extractor for the extension. If one
   * is present, it is invoked with the full file path and its output is
   * appended. Otherwise, a lightweight content_kind=... tag is emitted so
   * L2 still contributes searchable text.
   */
  void RunL2Dispatch(IndexPayload &out, const std::string &path,
                     const std::string &ext,
                     const std::vector<char> &data) const {
    // Copy the extractor under lock so we can invoke it without holding mu_
    // (some extractors may take seconds for large files; we don't want to
    // block other depth operations).
    L2Extractor fn;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = l2_extractors_.find(ext);
      if (it != l2_extractors_.end()) fn = it->second;
    }
    if (fn) {
      std::string rich = fn(path);
      if (!rich.empty()) {
        out.text += " ";
        out.text += rich;
        return;
      }
      // Registered extractor returned empty — fall through to the tag.
    }

    // Default: emit a content_kind=... marker for well-known scientific
    // formats. Richer extraction is plug-in territory.
    if (ext == "h5" || ext == "hdf5") {
      out.text += " content_kind=hdf5_scientific";
    } else if (ext == "nc" || ext == "nc4" || ext == "netcdf") {
      out.text += " content_kind=netcdf_scientific";
    } else if (ext == "parquet" || ext == "pq") {
      out.text += " content_kind=parquet_columnar";
    } else if (ext == "csv" || ext == "tsv") {
      out.text += " content_kind=tabular_text";
    }
    (void)data;
  }

  /** L3 — generate embedding from the accumulated L0-L2 text. */
  void RunL3(IndexPayload &out) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (!embedder_.Configured()) return;
    out.embedding = embedder_.Embed(out.text);
  }

  /** L4 — deep content analysis. Placeholder: statistical summary stub. */
  static void RunL4(IndexPayload &out, const std::string &ext,
                    const std::vector<char> &data) {
    // Extension point for per-format statistical summaries.
    // E.g. HDF5: read each dataset and compute min/max/mean.
    // For now, record a coarse signal (first/last/mid byte diversity).
    if (data.empty()) return;
    uint64_t n = data.size();
    uint64_t sum = 0;
    uint64_t sumsq = 0;
    uint64_t samples = 0;
    uint64_t stride = n > 4096 ? n / 1024 : 1;  // coarse sample
    for (uint64_t i = 0; i < n; i += stride) {
      uint8_t b = static_cast<uint8_t>(data[i]);
      sum += b;
      sumsq += static_cast<uint64_t>(b) * b;
      ++samples;
    }
    if (samples == 0) return;
    double mean = static_cast<double>(sum) / samples;
    double var = static_cast<double>(sumsq) / samples - mean * mean;
    out.text += " byte_mean=";
    out.text += std::to_string(mean);
    out.text += " byte_var=";
    out.text += std::to_string(var);
    (void)ext;
  }

  // ---------- Helpers ----------

  static std::string FileExtension(const std::string &name) {
    auto dot = name.find_last_of('.');
    if (dot == std::string::npos) return {};
    std::string ext = name.substr(dot + 1);
    for (auto &c : ext) c = static_cast<char>(std::tolower(c));
    return ext;
  }

  static std::string SniffFormat(const std::string &ext,
                                 const std::vector<char> &data) {
    // Extension hint first (cheap)
    if (!ext.empty()) return ext;
    // Magic-byte fallback when extension is missing
    if (data.size() >= 8) {
      static const unsigned char kHdf5Magic[8] = {0x89, 'H', 'D', 'F',
                                                  '\r', '\n', 0x1A, '\n'};
      if (std::memcmp(data.data(), kHdf5Magic, 8) == 0) return "hdf5";
    }
    if (data.size() >= 4) {
      if (data[0] == 'P' && data[1] == 'A' && data[2] == 'R' && data[3] == '1')
        return "parquet";
    }
    return {};
  }

#ifdef __linux__
  static std::optional<IndexDepth> ReadDepthXattr(const std::string &path,
                                                  const char *attr_name) {
    char buf[16] = {0};
    ssize_t n = ::getxattr(path.c_str(), attr_name, buf, sizeof(buf) - 1);
    if (n <= 0) return std::nullopt;
    int level = std::atoi(buf);
    if (level < 0 || level > 4) return std::nullopt;
    return static_cast<IndexDepth>(level);
  }
#endif

  mutable std::mutex mu_;
  DepthDefaults defaults_;
  EmbeddingClient embedder_;
  std::unordered_map<std::string, L2Extractor> l2_extractors_;
};

}  // namespace wrp_cte::core
#endif  // WRPCTE_DEPTH_CONTROLLER_H_
