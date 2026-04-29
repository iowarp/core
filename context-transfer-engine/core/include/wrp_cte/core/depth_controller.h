#ifndef WRPCTE_DEPTH_CONTROLLER_H_
#define WRPCTE_DEPTH_CONTROLLER_H_

/**
 * DepthController — Acropolis adaptive indexing depth.
 *
 * Three levels, additive (running at level N includes the work of all
 * lower levels):
 *
 *   L0 (Name)     filename, path, size, ext                  ~zero cost
 *   L1 (Metadata) + format detection + format-specific
 *                   metadata extraction (HDF5 dataset tree,
 *                   Parquet schema, etc.)
 *                 + embedding generated from the accumulated
 *                   text                                       ~ms + embedding
 *   L2 (Content)  same as L1 inside the controller, but the
 *                 caller is expected to have run an LLM
 *                 summary operator (e.g. CAE's SummaryOperator)
 *                 and pass the natural-language summary as the
 *                 `summary` argument to AsyncUpdateKnowledgeGraph;
 *                 the runtime handler concatenates that summary
 *                 onto the controller payload before storing.
 *                                                              ~LLM call
 *
 * In other words: the DepthController itself does L0 and L1. L2 is "L1
 * plus an LLM summary the caller provides." This keeps CTE free of any
 * LLM-calling code; the LLM lives in CAE's SummaryOperator.
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

/** What the controller feeds to the configured backend. */
struct IndexPayload {
  TagId tag_id;
  std::string text;                  ///< accumulated indexable text
  std::vector<float> embedding;      ///< populated at L1 and L2 when an
                                      ///< embedder is configured
  IndexDepth depth_achieved = IndexDepth::kNameOnly;
};

/** YAML-driven default levels per format extension. */
struct DepthDefaults {
  IndexDepth global_default = IndexDepth::kNameOnly;
  std::map<std::string, IndexDepth> per_format;  ///< key: lowercase extension
};

/**
 * MetadataExtractor — pluggable per-format metadata extractor invoked at L1.
 *
 * Takes a file path, returns a string describing the file's internal
 * structure (HDF5 dataset tree, Parquet schema, NetCDF variables, etc.).
 * The returned string is appended to the L1 payload text and indexed by
 * the configured backend.
 *
 * Implementations must be safe to call concurrently from multiple runtime
 * workers. They should degrade gracefully (return empty string) if the
 * file cannot be opened.
 */
using MetadataExtractor = std::function<std::string(const std::string &path)>;

class DepthController {
 public:
  DepthController() = default;

  /** Install a default-policy table loaded from YAML / config. */
  void SetDefaults(DepthDefaults defaults) {
    std::lock_guard<std::mutex> lk(mu_);
    defaults_ = std::move(defaults);
  }

  /** Configure the embedding client used for L1 and L2. */
  void SetEmbedder(EmbeddingClient embedder) {
    std::lock_guard<std::mutex> lk(mu_);
    embedder_ = std::move(embedder);
  }

  /**
   * Register a metadata extractor for a specific file extension. The
   * extension key is lowercase, without leading dot (e.g. "h5", "hdf5",
   * "parquet"). Invoked at L1 and L2.
   *
   * Later registrations for the same extension overwrite earlier ones.
   */
  void RegisterMetadataExtractor(const std::string &extension,
                                  MetadataExtractor fn) {
    std::lock_guard<std::mutex> lk(mu_);
    metadata_extractors_[extension] = std::move(fn);
  }

  /** Remove any registered extractor for a given extension. */
  void UnregisterMetadataExtractor(const std::string &extension) {
    std::lock_guard<std::mutex> lk(mu_);
    metadata_extractors_.erase(extension);
  }

  /** Query if a metadata extractor exists (mainly for tests). */
  bool HasMetadataExtractor(const std::string &extension) const {
    std::lock_guard<std::mutex> lk(mu_);
    return metadata_extractors_.count(extension) > 0;
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
   *
   *   L0 → just filename/size/ext text
   *   L1 → L0 text + format sniff + extractor output + embedding
   *   L2 → exactly the same as L1 inside the controller. The runtime
   *        handler appends the caller-supplied LLM summary on top and
   *        stores the combined string in the backend.
   *
   * `data` is unused today but kept on the interface for future content
   * inspection (e.g. statistical profiles).
   */
  IndexPayload Index(const TagId &tag_id,
                     const std::string &tag_name_or_path,
                     const std::vector<char> &data,
                     uint64_t file_size,
                     IndexDepth target) const {
    IndexPayload out;
    out.tag_id = tag_id;

    std::string ext = FileExtension(tag_name_or_path);

    RunL0(out, tag_name_or_path, file_size);
    out.depth_achieved = IndexDepth::kNameOnly;
    if (target == IndexDepth::kNameOnly) return out;

    // L1 and L2 do the same work inside the controller. The L2 contract
    // adds the caller-supplied LLM summary at the runtime layer.
    RunL1(out, tag_name_or_path, ext, data);
    EmbedIfConfigured(out);
    out.depth_achieved = target;  // record the level the caller asked for
    return out;
  }

 private:
  // ---------- Level executors (implemented inline, header-only) ----------

  static void RunL0(IndexPayload &out, const std::string &name,
                    uint64_t file_size) {
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

  /**
   * L1 — format sniff + format-specific extractor (if registered) + a
   * lightweight content_kind=... fallback for known scientific formats
   * when no extractor is available.
   */
  void RunL1(IndexPayload &out, const std::string &path,
             const std::string &ext,
             const std::vector<char> &data) const {
    // Format sniffing
    std::string fmt = SniffFormat(ext, data);
    if (!fmt.empty()) {
      out.text += " format=";
      out.text += fmt;
    }

    // Registered metadata extractor
    MetadataExtractor fn;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = metadata_extractors_.find(ext);
      if (it != metadata_extractors_.end()) fn = it->second;
    }
    if (fn) {
      std::string rich = fn(path);
      if (!rich.empty()) {
        out.text += " ";
        out.text += rich;
        return;  // extractor's output replaces the content_kind fallback
      }
    }

    // Default fallback: lightweight content_kind tag
    if (ext == "h5" || ext == "hdf5") {
      out.text += " content_kind=hdf5_scientific";
    } else if (ext == "nc" || ext == "nc4" || ext == "netcdf") {
      out.text += " content_kind=netcdf_scientific";
    } else if (ext == "parquet" || ext == "pq") {
      out.text += " content_kind=parquet_columnar";
    } else if (ext == "csv" || ext == "tsv") {
      out.text += " content_kind=tabular_text";
    }
  }

  /** Generate an embedding from the accumulated text (L1 and L2 only). */
  void EmbedIfConfigured(IndexPayload &out) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (!embedder_.Configured()) return;
    out.embedding = embedder_.Embed(out.text);
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
    if (!ext.empty()) return ext;
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
    if (level < 0 || level > 2) return std::nullopt;
    return static_cast<IndexDepth>(level);
  }
#endif

  mutable std::mutex mu_;
  DepthDefaults defaults_;
  EmbeddingClient embedder_;
  std::unordered_map<std::string, MetadataExtractor> metadata_extractors_;
};

}  // namespace wrp_cte::core
#endif  // WRPCTE_DEPTH_CONTROLLER_H_
