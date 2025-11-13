#include <wrp_cee/api/context_interface.h>
#include <wrp_cae/core/core_client.h>
#include <wrp_cte/core/core_client.h>
#include <wrp_cae/core/constants.h>
#include <chimaera/chimaera.h>
#include <iostream>

namespace iowarp {

ContextInterface::ContextInterface() : is_initialized_(false) {
  // Initialize Chimaera client if not already initialized
  if (!chi::CHIMAERA_CLIENT_INIT()) {
    std::cerr << "Error: Failed to initialize Chimaera client" << std::endl;
    return;
  }

  // Verify Chimaera IPC is available
  auto* ipc_manager = CHI_IPC;
  if (!ipc_manager) {
    std::cerr << "Error: Chimaera IPC not initialized. Is the runtime running?" << std::endl;
    return;
  }

  is_initialized_ = true;
}

ContextInterface::~ContextInterface() {
  // Cleanup if needed
}

int ContextInterface::ContextBundle(
    const std::vector<wrp_cae::core::AssimilationCtx> &bundle) {
  if (!is_initialized_) {
    std::cerr << "Error: ContextInterface not initialized" << std::endl;
    return 1;
  }

  if (bundle.empty()) {
    std::cerr << "Warning: Empty bundle provided to ContextBundle" << std::endl;
    return 0;
  }

  try {
    // Connect to CAE core container using the standard pool ID
    wrp_cae::core::Client cae_client(wrp_cae::core::kCaePoolId);

    // Call ParseOmni with vector of contexts
    chi::u32 num_tasks_scheduled = 0;
    chi::u32 result = cae_client.ParseOmni(HSHM_MCTX, bundle, num_tasks_scheduled);

    if (result != 0) {
      std::cerr << "Error: ParseOmni failed with result code " << result << std::endl;
      return static_cast<int>(result);
    }

    std::cout << "ContextBundle completed successfully!" << std::endl;
    std::cout << "  Tasks scheduled: " << num_tasks_scheduled << std::endl;

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error in ContextBundle: " << e.what() << std::endl;
    return 1;
  }
}

std::vector<std::string> ContextInterface::ContextQuery(
    const std::string &tag_re,
    const std::string &blob_re) {
  if (!is_initialized_) {
    std::cerr << "Error: ContextInterface not initialized" << std::endl;
    return std::vector<std::string>();
  }

  try {
    // Get the CTE client singleton
    auto* cte_client = WRP_CTE_CLIENT;
    if (!cte_client) {
      std::cerr << "Error: CTE client not initialized" << std::endl;
      return std::vector<std::string>();
    }

    // Call BlobQuery with tag and blob regex patterns
    // Use Broadcast to query across all nodes
    std::vector<std::string> results = cte_client->BlobQuery(
        HSHM_MCTX,
        tag_re,
        blob_re,
        chi::PoolQuery::Broadcast());

    return results;

  } catch (const std::exception& e) {
    std::cerr << "Error in ContextQuery: " << e.what() << std::endl;
    return std::vector<std::string>();
  }
}

std::vector<std::string> ContextInterface::ContextRetrieve(
    const std::string &tag_re,
    const std::string &blob_re) {
  (void)tag_re;   // Suppress unused parameter warning
  (void)blob_re;  // Suppress unused parameter warning

  // Not yet implemented
  std::cerr << "Warning: ContextRetrieve is not yet implemented" << std::endl;
  return std::vector<std::string>();
}

int ContextInterface::ContextSplice(
    const std::string &new_ctx,
    const std::string &tag_re,
    const std::string &blob_re) {
  (void)new_ctx;  // Suppress unused parameter warning
  (void)tag_re;   // Suppress unused parameter warning
  (void)blob_re;  // Suppress unused parameter warning

  // Not yet implemented
  std::cerr << "Warning: ContextSplice is not yet implemented" << std::endl;
  return 1;
}

int ContextInterface::ContextDestroy(
    const std::vector<std::string> &context_names) {
  if (!is_initialized_) {
    std::cerr << "Error: ContextInterface not initialized" << std::endl;
    return 1;
  }

  if (context_names.empty()) {
    std::cerr << "Warning: Empty context_names list provided to ContextDestroy" << std::endl;
    return 0;
  }

  try {
    // Get the CTE client singleton
    auto* cte_client = WRP_CTE_CLIENT;
    if (!cte_client) {
      std::cerr << "Error: CTE client not initialized" << std::endl;
      return 1;
    }

    // Iterate over each context name and delete the corresponding tag
    int error_count = 0;
    for (const auto& context_name : context_names) {
      bool result = cte_client->DelTag(HSHM_MCTX, context_name);
      if (!result) {
        std::cerr << "Error: Failed to delete context '" << context_name << "'" << std::endl;
        error_count++;
      } else {
        std::cout << "Successfully deleted context: " << context_name << std::endl;
      }
    }

    if (error_count > 0) {
      std::cerr << "ContextDestroy completed with " << error_count << " error(s)" << std::endl;
      return 1;
    }

    std::cout << "ContextDestroy completed successfully!" << std::endl;
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error in ContextDestroy: " << e.what() << std::endl;
    return 1;
  }
}

}  // namespace iowarp
