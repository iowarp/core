#ifndef WRPCTE_KG_BACKEND_FACTORY_H_
#define WRPCTE_KG_BACKEND_FACTORY_H_

#include <wrp_cte/core/kg_backend.h>
#include <wrp_cte/core/kg_backend_bm25.h>

#ifdef WRP_CTE_KG_ELASTICSEARCH
#include <wrp_cte/core/kg_backend_elasticsearch.h>
#endif
#ifdef WRP_CTE_KG_NEO4J
#include <wrp_cte/core/kg_backend_neo4j.h>
#endif
#ifdef WRP_CTE_KG_QDRANT
#include <wrp_cte/core/kg_backend_qdrant.h>
#endif

#include <memory>
#include <string>

namespace wrp_cte::core {

/**
 * Create a KGBackend by name. Returns BM25 as default.
 * Additional backends are enabled via compile-time flags.
 */
inline std::unique_ptr<KGBackend> CreateKGBackend(const std::string &type) {
  if (type.empty() || type == "bm25") {
    return std::make_unique<BM25Backend>();
  }
#ifdef WRP_CTE_KG_ELASTICSEARCH
  if (type == "elasticsearch") return std::make_unique<ElasticsearchBackend>();
#endif
#ifdef WRP_CTE_KG_NEO4J
  if (type == "neo4j") return std::make_unique<Neo4jBackend>();
#endif
#ifdef WRP_CTE_KG_QDRANT
  if (type == "qdrant") return std::make_unique<QdrantBackend>();
#endif

  // Unknown type — fall back to BM25
  return std::make_unique<BM25Backend>();
}

}  // namespace wrp_cte::core
#endif  // WRPCTE_KG_BACKEND_FACTORY_H_
