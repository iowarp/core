/**
 * Pool query implementation
 */

#include "chimaera/pool_query.h"
#include <algorithm>
#include <stdexcept>

namespace chi {

PoolQuery::PoolQuery()
    : routing_mode_(RoutingMode::Local), hash_value_(0), container_id_(0),
      range_offset_(0), range_count_(0), node_id_(0), ret_node_(0) {}

PoolQuery::PoolQuery(const PoolQuery& other)
    : routing_mode_(other.routing_mode_),
      hash_value_(other.hash_value_),
      container_id_(other.container_id_),
      range_offset_(other.range_offset_),
      range_count_(other.range_count_),
      node_id_(other.node_id_),
      ret_node_(other.ret_node_) {}

PoolQuery& PoolQuery::operator=(const PoolQuery& other) {
  if (this != &other) {
    routing_mode_ = other.routing_mode_;
    hash_value_ = other.hash_value_;
    container_id_ = other.container_id_;
    range_offset_ = other.range_offset_;
    range_count_ = other.range_count_;
    node_id_ = other.node_id_;
    ret_node_ = other.ret_node_;
  }
  return *this;
}

PoolQuery::~PoolQuery() {
  // Stub destructor
}

// Static factory methods

PoolQuery PoolQuery::Local() {
  PoolQuery query;
  query.routing_mode_ = RoutingMode::Local;
  query.hash_value_ = 0;
  query.container_id_ = 0;
  query.range_offset_ = 0;
  query.range_count_ = 0;
  query.node_id_ = 0;
  return query;
}

PoolQuery PoolQuery::DirectId(ContainerId container_id) {
  PoolQuery query;
  query.routing_mode_ = RoutingMode::DirectId;
  query.hash_value_ = 0;
  query.container_id_ = container_id;
  query.range_offset_ = 0;
  query.range_count_ = 0;
  query.node_id_ = 0;
  return query;
}

PoolQuery PoolQuery::DirectHash(u32 hash) {
  PoolQuery query;
  query.routing_mode_ = RoutingMode::DirectHash;
  query.hash_value_ = hash;
  query.container_id_ = 0;
  query.range_offset_ = 0;
  query.range_count_ = 0;
  query.node_id_ = 0;
  return query;
}

PoolQuery PoolQuery::Range(u32 offset, u32 count) {
  PoolQuery query;
  query.routing_mode_ = RoutingMode::Range;
  query.hash_value_ = 0;
  query.container_id_ = 0;
  query.range_offset_ = offset;
  query.range_count_ = count;
  query.node_id_ = 0;
  return query;
}

PoolQuery PoolQuery::Broadcast() {
  PoolQuery query;
  query.routing_mode_ = RoutingMode::Broadcast;
  query.hash_value_ = 0;
  query.container_id_ = 0;
  query.range_offset_ = 0;
  query.range_count_ = 0;
  query.node_id_ = 0;
  return query;
}

PoolQuery PoolQuery::Physical(u32 node_id) {
  PoolQuery query;
  query.routing_mode_ = RoutingMode::Physical;
  query.hash_value_ = 0;
  query.container_id_ = 0;
  query.range_offset_ = 0;
  query.range_count_ = 0;
  query.node_id_ = node_id;
  return query;
}

PoolQuery PoolQuery::Dynamic() {
  PoolQuery query;
  query.routing_mode_ = RoutingMode::Dynamic;
  query.hash_value_ = 0;
  query.container_id_ = 0;
  query.range_offset_ = 0;
  query.range_count_ = 0;
  query.node_id_ = 0;
  return query;
}

PoolQuery PoolQuery::FromString(const std::string& str) {
  // Convert to lowercase for case-insensitive comparison
  std::string lower_str = str;
  std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (lower_str == "local") {
    return PoolQuery::Local();
  } else if (lower_str == "dynamic") {
    return PoolQuery::Dynamic();
  } else {
    throw std::invalid_argument("Invalid PoolQuery string, expected 'local' or 'dynamic'");
  }
}

// Getter methods

u32 PoolQuery::GetHash() const { return hash_value_; }

ContainerId PoolQuery::GetContainerId() const { return container_id_; }

u32 PoolQuery::GetRangeOffset() const { return range_offset_; }

u32 PoolQuery::GetRangeCount() const { return range_count_; }

u32 PoolQuery::GetNodeId() const { return node_id_; }

RoutingMode PoolQuery::GetRoutingMode() const { return routing_mode_; }

bool PoolQuery::IsLocalMode() const {
  return routing_mode_ == RoutingMode::Local;
}

bool PoolQuery::IsDirectIdMode() const {
  return routing_mode_ == RoutingMode::DirectId;
}

bool PoolQuery::IsDirectHashMode() const {
  return routing_mode_ == RoutingMode::DirectHash;
}

bool PoolQuery::IsRangeMode() const {
  return routing_mode_ == RoutingMode::Range;
}

bool PoolQuery::IsBroadcastMode() const {
  return routing_mode_ == RoutingMode::Broadcast;
}

bool PoolQuery::IsPhysicalMode() const {
  return routing_mode_ == RoutingMode::Physical;
}

bool PoolQuery::IsDynamicMode() const {
  return routing_mode_ == RoutingMode::Dynamic;
}

void PoolQuery::SetReturnNode(u32 ret_node) {
  ret_node_ = ret_node;
}

u32 PoolQuery::GetReturnNode() const {
  return ret_node_;
}

}  // namespace chi