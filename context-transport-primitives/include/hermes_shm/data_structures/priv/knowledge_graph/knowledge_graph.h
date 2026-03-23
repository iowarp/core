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

#ifndef HSHM_DATA_STRUCTURES_PRIV_KNOWLEDGE_GRAPH_H_
#define HSHM_DATA_STRUCTURES_PRIV_KNOWLEDGE_GRAPH_H_

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace hshm::priv {

/**
 * Result of a knowledge graph search operation.
 * @tparam Key The key type used to identify entries
 */
template <typename Key>
struct SearchResult {
  Key key;
  float score;

  bool operator<(const SearchResult &other) const {
    return score > other.score;  // Descending order (highest score first)
  }
};

/**
 * A single entry in the knowledge graph.
 * Stores the key, value text, and pre-computed term frequencies.
 */
template <typename Key>
struct KnowledgeGraphEntry {
  Key key;
  std::string value;
  std::unordered_map<std::string, size_t> term_counts;
  size_t total_terms = 0;
};

/**
 * STL-based knowledge graph with BM25 text search.
 *
 * Stores key-value pairs where values are text strings.
 * Provides semantic-like search via BM25 ranking over tokenized text.
 * Uses an inverted index for efficient query evaluation.
 *
 * @tparam Key The key type (must be hashable)
 * @tparam Hash Hash function for the key type
 */
template <typename Key, typename Hash = std::hash<Key>>
class KnowledgeGraph {
 public:
  /**
   * Add a key-value pair to the knowledge graph.
   * If the key already exists, the value is updated.
   * @param key Unique identifier for the entry
   * @param value Text content to store and index
   */
  void Add(const Key &key, const std::string &value) {
    auto it = key_to_idx_.find(key);
    if (it != key_to_idx_.end()) {
      // Update existing entry: remove old index, replace
      RemoveFromIndex(it->second);
      auto &entry = entries_[it->second];
      entry.value = value;
      entry.term_counts.clear();
      entry.total_terms = 0;
      Tokenize(value, entry.term_counts, entry.total_terms);
      AddToIndex(it->second);
      return;
    }

    // New entry
    size_t idx = entries_.size();
    entries_.push_back({key, value, {}, 0});
    auto &entry = entries_.back();
    Tokenize(value, entry.term_counts, entry.total_terms);
    key_to_idx_[key] = idx;
    AddToIndex(idx);
  }

  /**
   * Get local document frequency counts for all terms.
   * Used to build global IDF by aggregating across nodes.
   * @return Map of term -> local document frequency
   */
  std::unordered_map<std::string, size_t> GetLocalDf() const {
    std::unordered_map<std::string, size_t> local_df;
    for (const auto &[term, doc_set] : inverted_index_) {
      local_df[term] = doc_set.size();
    }
    return local_df;
  }

  /** Get the number of local documents */
  size_t GetLocalN() const { return entries_.size(); }

  /** Get the total term count across all local documents */
  size_t GetLocalTotalTerms() const {
    size_t total = 0;
    for (const auto &e : entries_) {
      total += e.total_terms;
    }
    return total;
  }

  /**
   * Set global IDF statistics collected from all nodes.
   * When set, Search() uses these instead of local stats.
   * @param global_n Total document count across all nodes
   * @param global_df Per-term document frequency across all nodes
   * @param global_avg_dl Average document length across all nodes
   */
  void SetGlobalIdf(size_t global_n,
                    std::unordered_map<std::string, size_t> global_df,
                    float global_avg_dl) {
    global_n_ = global_n;
    global_df_ = std::move(global_df);
    global_avg_dl_ = global_avg_dl;
    use_global_idf_ = true;
  }

  /**
   * Search the knowledge graph for entries matching a text prompt.
   * Uses BM25 ranking to score entries against the query.
   * When global IDF stats are set (via SetGlobalIdf), uses global N and df
   * for IDF calculation so scores are comparable across distributed nodes.
   * @param prompt The search query text
   * @param top_k Maximum number of results to return
   * @return Vector of SearchResult sorted by descending confidence score
   */
  std::vector<SearchResult<Key>> Search(const std::string &prompt,
                                        int top_k) const {
    std::unordered_map<std::string, size_t> query_terms;
    size_t query_total;
    Tokenize(prompt, query_terms, query_total);

    if (query_terms.empty() || entries_.empty()) {
      return {};
    }

    // Use global stats if available, otherwise local
    const size_t N = use_global_idf_ ? global_n_ : entries_.size();
    const float avg_dl = use_global_idf_ ? global_avg_dl_ : ComputeAvgDocLen();

    // Score each entry using BM25
    std::vector<SearchResult<Key>> results;
    results.reserve(entries_.size());

    for (size_t i = 0; i < entries_.size(); ++i) {
      if (entries_[i].total_terms == 0) continue;

      float score = 0.0f;
      for (const auto &[term, _] : query_terms) {
        // Check if this entry contains the term (local TF)
        auto tc_it = entries_[i].term_counts.find(term);
        if (tc_it == entries_[i].term_counts.end()) continue;

        // Get df: global if available, otherwise local
        size_t df = 0;
        if (use_global_idf_) {
          auto gdf_it = global_df_.find(term);
          if (gdf_it != global_df_.end()) {
            df = gdf_it->second;
          }
        } else {
          auto inv_it = inverted_index_.find(term);
          if (inv_it != inverted_index_.end()) {
            df = inv_it->second.size();
          }
        }
        if (df == 0) continue;

        // IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        float idf = std::log(
            (static_cast<float>(N) - static_cast<float>(df) + 0.5f) /
                (static_cast<float>(df) + 0.5f) +
            1.0f);

        float tf = static_cast<float>(tc_it->second);
        float dl = static_cast<float>(entries_[i].total_terms);

        // BM25: idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
        float numerator = tf * (kK1 + 1.0f);
        float denominator = tf + kK1 * (1.0f - kB + kB * dl / avg_dl);
        score += idf * numerator / denominator;
      }

      if (score > 0.0f) {
        results.push_back({entries_[i].key, score});
      }
    }

    // Sort by score descending
    std::sort(results.begin(), results.end());

    // Truncate to top_k
    if (top_k > 0 && results.size() > static_cast<size_t>(top_k)) {
      results.resize(static_cast<size_t>(top_k));
    }

    return results;
  }

  /**
   * Remove an entry by key.
   * @param key Key of the entry to remove
   */
  void Remove(const Key &key) {
    auto it = key_to_idx_.find(key);
    if (it == key_to_idx_.end()) return;

    size_t idx = it->second;
    RemoveFromIndex(idx);

    // Swap-and-pop for O(1) removal
    size_t last = entries_.size() - 1;
    if (idx != last) {
      // Update the moved entry's index
      key_to_idx_[entries_[last].key] = idx;
      RemoveFromIndex(last);
      entries_[idx] = std::move(entries_[last]);
      AddToIndex(idx);
    }
    entries_.pop_back();
    key_to_idx_.erase(it);
  }

  /** Return the number of entries in the graph */
  size_t Size() const { return entries_.size(); }

  /** Remove all entries */
  void Clear() {
    entries_.clear();
    key_to_idx_.clear();
    inverted_index_.clear();
  }

 private:
  // BM25 parameters
  static constexpr float kK1 = 1.2f;
  static constexpr float kB = 0.75f;

  // Storage
  std::vector<KnowledgeGraphEntry<Key>> entries_;
  std::unordered_map<Key, size_t, Hash> key_to_idx_;
  std::unordered_map<std::string, std::unordered_set<size_t>> inverted_index_;

  // Global IDF stats (set by SetGlobalIdf after distributed sync)
  bool use_global_idf_ = false;
  size_t global_n_ = 0;
  std::unordered_map<std::string, size_t> global_df_;
  float global_avg_dl_ = 1.0f;

  /**
   * Tokenize text into lowercase words, updating term counts.
   */
  static void Tokenize(const std::string &text,
                        std::unordered_map<std::string, size_t> &term_counts,
                        size_t &total_terms) {
    total_terms = 0;
    std::string word;
    for (char c : text) {
      if (std::isalnum(static_cast<unsigned char>(c))) {
        word += static_cast<char>(
            std::tolower(static_cast<unsigned char>(c)));
      } else if (!word.empty()) {
        if (word.size() > 1) {  // Skip single-char tokens
          term_counts[word]++;
          total_terms++;
        }
        word.clear();
      }
    }
    if (!word.empty() && word.size() > 1) {
      term_counts[word]++;
      total_terms++;
    }
  }

  /** Add entry's terms to the inverted index */
  void AddToIndex(size_t idx) {
    for (const auto &[term, _] : entries_[idx].term_counts) {
      inverted_index_[term].insert(idx);
    }
  }

  /** Remove entry's terms from the inverted index */
  void RemoveFromIndex(size_t idx) {
    for (const auto &[term, _] : entries_[idx].term_counts) {
      auto it = inverted_index_.find(term);
      if (it != inverted_index_.end()) {
        it->second.erase(idx);
        if (it->second.empty()) {
          inverted_index_.erase(it);
        }
      }
    }
  }

  /** Compute average document length across all entries */
  float ComputeAvgDocLen() const {
    if (entries_.empty()) return 1.0f;
    size_t total = 0;
    for (const auto &e : entries_) {
      total += e.total_terms;
    }
    return static_cast<float>(total) / static_cast<float>(entries_.size());
  }
};

}  // namespace hshm::priv

#endif  // HSHM_DATA_STRUCTURES_PRIV_KNOWLEDGE_GRAPH_H_
