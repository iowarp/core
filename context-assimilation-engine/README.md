# Context Assimilation Engine (CAE)

A Chimaera module (ChiMod) for high-performance data ingestion into the IOWarp
ecosystem. CAE assimilates data from external sources — local binary files,
HDF5 datasets, and Globus endpoints — into the Context Transfer Engine (CTE)
for distributed storage and retrieval.

## Overview

CAE runs as a pool inside the Chimaera runtime alongside CTE. Clients submit
OMNI YAML files to describe data transfers; CAE parses them and dispatches
assimilation tasks to the appropriate backend (binary, HDF5, or Globus).

```
External Source          CAE ChiMod              CTE ChiMod
(file, HDF5, Globus) --> ParseOmni task -------> Tag + Blob storage
```

**Supported formats:** `binary`, `hdf5`, `globus`

## Building

CAE is built as part of the IOWarp Core monorepo:

```bash
git clone https://github.com/iowarp/clio-core.git
cd clio-core
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

To enable HDF5 support:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DWRP_CORE_ENABLE_HDF5=ON ..
```

## Running

### 1. Start the Chimaera runtime with CTE and CAE

```bash
export CHI_SERVER_CONF=/path/to/wrp_config.yaml
chimaera runtime start
```

An example configuration deploying both CTE and CAE is provided in
`config/wrp_config_example.yaml`:

```yaml
compose:
  - mod_name: wrp_cte_core
    pool_name: cte_main
    pool_query: local
    pool_id: "512.0"
  - mod_name: wrp_cae_core
    pool_name: cae_main
    pool_query: local
    pool_id: "400.0"
```

### 2. Submit an OMNI file

```bash
wrp_cae_omni /path/to/my_transfers.yaml
```

## OMNI File Format

OMNI files describe data transfers in YAML using a `transfers` list:

```yaml
name: my_ingestion_job

transfers:
  - src: file::/path/to/data.bin    # Source URL (required)
    dst: iowarp::my_tag             # CTE destination tag (required)
    format: binary                  # Format: binary, hdf5, globus (required)
    depends_on: ""                  # Dependency identifier (optional)
    range_off: 0                    # Byte offset in source (optional)
    range_size: 0                   # Bytes to read, 0 = full file (optional)
    src_token: $GLOBUS_TOKEN        # Auth token, env vars expanded (optional)
```

**HDF5 with dataset filtering:**
```yaml
transfers:
  - src: file::/data/experiment.h5
    dst: iowarp::experiment
    format: hdf5
    dataset_filter:
      include_patterns:
        - "/sensors/*"
      exclude_patterns:
        - "/sensors/raw/*"
```

**Key fields:**
| Field | Description |
|---|---|
| `src` | Source URL (`file::`, `globus::`) |
| `dst` | CTE destination tag (`iowarp::tag_name`) |
| `format` | `binary`, `hdf5`, or `globus` |
| `range_off` / `range_size` | Byte range within source (0 = full file) |
| `src_token` / `dst_token` | Auth tokens; environment variables are expanded |
| `dataset_filter` | HDF5 dataset include/exclude glob patterns |

## Project Structure

```
config/      - Example runtime configuration files
core/        - ChiMod implementation
  include/   - Public headers (tasks, assimilation context, factory)
  src/        - Runtime and client implementation
  util/       - wrp_cae_omni command-line tool
data/        - Sample datasets for testing (HDF5, CSV, Parquet)
test/
  unit/      - Unit tests (binary, HDF5, range, error handling)
  integration/globus_matsci/ - Globus integration test
```

## Pool ID Reference

| Component | Pool ID | Module Name    |
|-----------|---------|----------------|
| Admin     | 1.0     | chimaera_admin |
| CTE Core  | 512.0   | wrp_cte_core   |
| CAE Core  | 400.0   | wrp_cae_core   |

## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2024, Gnosis Research Center, Illinois Institute of Technology**

## Links

- **IOWarp Organization**: [https://github.com/iowarp](https://github.com/iowarp)
- **Issues**: [https://github.com/iowarp/clio-core/issues](https://github.com/iowarp/clio-core/issues)
