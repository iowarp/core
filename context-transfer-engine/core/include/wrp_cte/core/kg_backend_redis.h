#ifndef WRPCTE_KG_BACKEND_REDIS_H_
#define WRPCTE_KG_BACKEND_REDIS_H_

#include <wrp_cte/core/kg_backend.h>
#include <hiredis/hiredis.h>
#include <string>
#include <sstream>
#include <cstdlib>

namespace wrp_cte::core {

/**
 * Redis + RediSearch backend — in-memory KV store with full-text search.
 * Architecturally closest to CTE (in-memory, key-value oriented).
 * Uses hiredis C client library.
 */
class RedisBackend : public KGBackend {
 public:
  std::string Name() const override { return "redis"; }

  void Init(const std::string &config) override {
    std::string host = "127.0.0.1";
    int port = 6379;

    if (!config.empty()) {
      auto colon = config.find(':');
      if (colon != std::string::npos) {
        host = config.substr(0, colon);
        port = std::stoi(config.substr(colon + 1));
      } else {
        host = config;
      }
    }

    ctx_ = redisConnect(host.c_str(), port);
    if (!ctx_ || ctx_->err) {
      if (ctx_) { redisFree(ctx_); ctx_ = nullptr; }
      return;
    }

    // Drop existing index if any, then create fresh
    {
      redisReply *r = static_cast<redisReply*>(
          redisCommand(ctx_, "FT.DROPINDEX cte_kg DD"));
      if (r) freeReplyObject(r);
    }
    {
      redisReply *r = static_cast<redisReply*>(
          redisCommand(ctx_,
              "FT.CREATE cte_kg ON HASH PREFIX 1 tag: "
              "SCHEMA text TEXT WEIGHT 1.0 tag_major NUMERIC tag_minor NUMERIC"));
      if (r) freeReplyObject(r);
    }
    size_ = 0;
  }

  void Destroy() override {
    if (ctx_) {
      Cmd("FT.DROPINDEX %s DD", "cte_kg");
      redisFree(ctx_);
      ctx_ = nullptr;
    }
  }

  void Add(const TagId &tag_id, const std::string &text) override {
    if (!ctx_) return;
    std::string key = "tag:" + std::to_string(tag_id.major_) + ":" +
                      std::to_string(tag_id.minor_);
    redisReply *reply = static_cast<redisReply*>(redisCommand(
        ctx_, "HSET %s text %s tag_major %u tag_minor %u",
        key.c_str(), text.c_str(), tag_id.major_, tag_id.minor_));
    if (reply) { freeReplyObject(reply); size_++; }
  }

  void Remove(const TagId &tag_id) override {
    if (!ctx_) return;
    std::string key = "tag:" + std::to_string(tag_id.major_) + ":" +
                      std::to_string(tag_id.minor_);
    redisReply *reply = static_cast<redisReply*>(
        redisCommand(ctx_, "DEL %s", key.c_str()));
    if (reply) { freeReplyObject(reply); if (size_ > 0) size_--; }
  }

  std::vector<KGSearchResult> Search(
      const std::string &query, int top_k) override {
    std::vector<KGSearchResult> results;
    if (!ctx_) return results;

    // RediSearch treats multi-word queries as exact phrases by default.
    // Split into individual words joined by | (OR) for term-level matching.
    std::string or_query;
    std::istringstream iss(query);
    std::string word;
    while (iss >> word) {
      if (!or_query.empty()) or_query += "|";
      or_query += word;
    }
    if (or_query.empty()) or_query = query;

    redisReply *reply = static_cast<redisReply*>(redisCommand(
        ctx_, "FT.SEARCH cte_kg %s LIMIT 0 %d WITHSCORES",
        or_query.c_str(), top_k));

    if (!reply || reply->type != REDIS_REPLY_ARRAY || reply->elements < 1) {
      if (reply) freeReplyObject(reply);
      return results;
    }

    // FT.SEARCH WITHSCORES format (hiredis 1.x):
    //   [total, key1, score1, [field1, val1, ...], key2, score2, [...], ...]
    // Each result: key (string), score (string), fields (array)
    size_t i = 1;  // skip total count
    while (i + 2 < reply->elements &&
           results.size() < static_cast<size_t>(top_k)) {
      // key at i (skip)
      // score at i+1
      float score = 0;
      if (reply->element[i + 1]->str) {
        try {
          score = std::stof(std::string(
              reply->element[i + 1]->str, reply->element[i + 1]->len));
        } catch (...) {}
      }

      // fields at i+2 (may be nested array or flat pairs)
      TagId tid;
      redisReply *fields = reply->element[i + 2];
      if (fields->type == REDIS_REPLY_ARRAY) {
        // Nested array: [field1, val1, field2, val2, ...]
        for (size_t j = 0; j + 1 < fields->elements; j += 2) {
          if (!fields->element[j]->str || !fields->element[j + 1]->str)
            continue;
          std::string fname(fields->element[j]->str, fields->element[j]->len);
          std::string fval(fields->element[j + 1]->str, fields->element[j + 1]->len);
          if (fname == "tag_major") tid.major_ = std::stoul(fval);
          if (fname == "tag_minor") tid.minor_ = std::stoul(fval);
        }
      }
      results.push_back({tid, score});
      i += 3;
    }

    freeReplyObject(reply);
    return results;
  }

  size_t Size() const override { return size_; }

  void Clear() override {
    if (ctx_) {
      Init("");  // Re-initialize drops and recreates
    }
  }

 private:
  template <typename... Args>
  void Cmd(const char *fmt, Args... args) {
    if (!ctx_) return;
    redisReply *r = static_cast<redisReply*>(
        redisCommand(ctx_, fmt, args...));
    if (r) freeReplyObject(r);
  }

  redisContext *ctx_ = nullptr;
  size_t size_ = 0;
};

}  // namespace wrp_cte::core
#endif  // WRPCTE_KG_BACKEND_REDIS_H_
