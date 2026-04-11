# Globus Materials Science Integration Test Result

**Date:** 2026-04-07  
**Status:** PASSED

## Configuration

| Parameter | Value |
|---|---|
| Collection ID | `82f1b5c6-6e9b-11e5-ba47-22000b92c6ec` |
| Collection | AFRL Challenge Data - Publication 1151 |
| Runtime Config | `context-assimilation-engine/test/integration/globus_matsci/wrp_runtime_conf.yaml` |
| OMNI File | `context-assimilation-engine/test/integration/globus_matsci/matsci_globus_omni.yaml` |
| Build Flags | `-DWRP_CORE_ENABLE_CAE=ON -DCAE_ENABLE_GLOBUS=ON` |

## Transferred Files

| File | Size | Destination |
|---|---|---|
| `HomeIn-Build B.csv` | 30K | `/tmp/globus_matsci/HomeIn-Build_B.csv` |
| `Build B Question Item Conditions.xlsx` | 9.9K | `/tmp/globus_matsci/Build_B_Question_Item_Conditions.xlsx` |

## Fixes Applied

1. **`run_test.sh`**: Changed `CHIMAERA_CONF` → `CHI_SERVER_CONF` so runtime loads the compose config.
2. **`wrp_runtime_conf.yaml`**: Added `chimaera_bdev` entry (required before CTE/CAE pools can initialize); reduced memory from 16GB/2GB to 256MB for login node compatibility.
3. **`run_test.sh`**: Auto-sources `/tmp/globus_tokens.sh` to export `GLOBUS_HTTPS_ACCESS_TOKEN`.
4. **`matsci_globus_omni.yaml`**: Switched source collection to AFRL (`82f1b5c6...`) which allows public HTTPS access.
5. **`get_oauth_token.py`**: Added `--auth-code` argument with PKCE state save/restore to avoid regenerating the challenge on each invocation.
