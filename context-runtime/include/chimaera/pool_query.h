#ifndef CHIMAERA_INCLUDE_CHIMAERA_POOL_QUERY_H_
#define CHIMAERA_INCLUDE_CHIMAERA_POOL_QUERY_H_

#include <cereal/cereal.hpp>

#include "chimaera/types.h"

namespace chi {

/**
 * Routing algorithm modes for PoolQuery
 */
enum class RoutingMode {
  Local,      /**< Route to local node only */
  DirectId,   /**< Route to specific container by ID */
  DirectHash, /**< Route using hash-based load balancing */
  Range,      /**< Route to range of containers */
  Broadcast,  /**< Broadcast to all containers */
  Physical,   /**< Route to specific physical node by ID */
  Dynamic     /**< Dynamic routing with cache optimization (routes to Monitor) */
};

/**
 * Pool query class for determining task execution location and routing
 *
 * Provides methods to query different container addresses and routing modes
 * for load balancing and task distribution to containers.
 */
class PoolQuery {
 public:
  /**
   * Default constructor
   */
  PoolQuery();

  /**
   * Copy constructor
   */
  PoolQuery(const PoolQuery& other);

  /**
   * Assignment operator
   */
  PoolQuery& operator=(const PoolQuery& other);

  /**
   * Destructor
   */
  ~PoolQuery();

  // Static factory methods to create different types of PoolQuery

  /**
   * Create a local routing pool query
   * @return PoolQuery configured for local container routing
   */
  static PoolQuery Local();

  /**
   * Create a direct ID routing pool query
   * @param container_id Specific container ID to route to
   * @return PoolQuery configured for direct container ID routing
   */
  static PoolQuery DirectId(ContainerId container_id);

  /**
   * Create a direct hash routing pool query
   * @param hash Hash value for container selection
   * @return PoolQuery configured for hash-based routing to specific container
   */
  static PoolQuery DirectHash(u32 hash);

  /**
   * Create a range routing pool query
   * @param offset Starting offset in the container range
   * @param count Number of containers in the range
   * @return PoolQuery configured for range-based routing
   */
  static PoolQuery Range(u32 offset, u32 count);

  /**
   * Create a broadcast routing pool query
   * @return PoolQuery configured for broadcast to all containers
   */
  static PoolQuery Broadcast();

  /**
   * Create a physical routing pool query
   * @param node_id Specific node ID to route to
   * @return PoolQuery configured for physical node routing
   */
  static PoolQuery Physical(u32 node_id);

  /**
   * Create a dynamic routing pool query (recommended for Create operations)
   * Routes to Monitor with kGlobalSchedule for automatic cache checking
   * @return PoolQuery configured for dynamic routing with cache optimization
   */
  static PoolQuery Dynamic();

  /**
   * Parse PoolQuery from string (supports "local" and "dynamic")
   * @param str String representation of pool query mode
   * @return PoolQuery configured based on string value
   */
  static PoolQuery FromString(const std::string& str);

  // Getter methods for internal query parameters (used by routing logic)

  /**
   * Get the hash value for hash-based routing modes
   * @return Hash value used for container routing
   */
  u32 GetHash() const;

  /**
   * Get the container ID for direct ID routing mode
   * @return Container ID for direct routing
   */
  ContainerId GetContainerId() const;

  /**
   * Get the range offset for range routing mode
   * @return Starting offset in the container range
   */
  u32 GetRangeOffset() const;

  /**
   * Get the range count for range routing mode
   * @return Number of containers in the range
   */
  u32 GetRangeCount() const;

  /**
   * Get the node ID for physical routing mode
   * @return Node ID for physical routing
   */
  u32 GetNodeId() const;

  /**
   * Determine the routing mode of this pool query
   * @return RoutingMode enum indicating how this query should be routed
   */
  RoutingMode GetRoutingMode() const;

  /**
   * Check if pool query is in Local routing mode
   * @return true if routing mode is Local
   */
  bool IsLocalMode() const;

  /**
   * Check if pool query is in DirectId routing mode
   * @return true if routing mode is DirectId
   */
  bool IsDirectIdMode() const;

  /**
   * Check if pool query is in DirectHash routing mode
   * @return true if routing mode is DirectHash
   */
  bool IsDirectHashMode() const;

  /**
   * Check if pool query is in Range routing mode
   * @return true if routing mode is Range
   */
  bool IsRangeMode() const;

  /**
   * Check if pool query is in Broadcast routing mode
   * @return true if routing mode is Broadcast
   */
  bool IsBroadcastMode() const;

  /**
   * Check if pool query is in Physical routing mode
   * @return true if routing mode is Physical
   */
  bool IsPhysicalMode() const;

  /**
   * Check if pool query is in Dynamic routing mode
   * @return true if routing mode is Dynamic
   */
  bool IsDynamicMode() const;

  /**
   * Set the return node ID for distributed task responses
   * @param ret_node Node ID where task results should be returned
   */
  void SetReturnNode(u32 ret_node);

  /**
   * Get the return node ID for distributed task responses
   * @return Node ID where task results should be returned
   */
  u32 GetReturnNode() const;

  /**
   * Cereal serialization support
   * @param ar Archive for serialization
   */
  template <class Archive>
  void serialize(Archive& ar) {
    ar(routing_mode_, hash_value_, container_id_, range_offset_, range_count_, node_id_, ret_node_);
  }

 private:
  RoutingMode routing_mode_; /**< The routing mode for this query */
  u32 hash_value_;           /**< Hash value for hash-based routing */
  ContainerId container_id_; /**< Container ID for direct ID routing */
  u32 range_offset_;         /**< Starting offset for range routing */
  u32 range_count_;          /**< Number of containers for range routing */
  u32 node_id_;              /**< Node ID for physical routing */
  u32 ret_node_;             /**< Return node ID for distributed responses */
};


}  // namespace chi

#endif  // CHIMAERA_INCLUDE_CHIMAERA_POOL_QUERY_H_