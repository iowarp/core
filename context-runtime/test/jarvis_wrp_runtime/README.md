# Jarvis IOWarp Runtime Package

This repository provides Jarvis packages for deploying and managing the IOWarp (Chimaera) runtime system across distributed nodes.

## Package Overview

### wrp_runtime (Service)

The `wrp_runtime` package deploys and manages the IOWarp runtime service across distributed nodes. It handles:

- Runtime configuration generation (creates `chimaera_config.yaml`)
- Setting the `CHI_SERVER_CONF` environment variable
- Distributed deployment using parallel SSH
- Service lifecycle management (start/stop/kill)
- Cleanup of shared memory and temporary files

**Assumes**: `chimaera_start_runtime` and `chimaera_stop_runtime` binaries are installed and available in `PATH`.

## Installation

Add this repository to Jarvis:

```bash
jarvis repo add /path/to/jarvis_wrp_runtime
```

## Usage

### Basic Deployment

```bash
# Load the basic runtime pipeline
jarvis ppl index load jarvis_wrp_runtime.basic_runtime

# Start the runtime on all nodes (uses default configuration)
jarvis ppl start

# Check runtime status
jarvis ppl status

# Stop the runtime
jarvis ppl stop

# Clean runtime data
jarvis ppl clean
```

### Custom Configuration

```bash
# Load the pipeline
jarvis ppl index load jarvis_wrp_runtime.basic_runtime

# Customize configuration parameters
jarvis pkg conf runtime low_latency_workers=8
jarvis pkg conf runtime main_segment_size=2G
jarvis pkg conf runtime log_level=debug

# Start with custom configuration
jarvis ppl start
```

### Configuration Parameters

#### Worker Threads
- **low_latency_workers**: Number of low-latency worker threads (default: `4`)
- **high_latency_workers**: Number of high-latency worker threads (default: `2`)
- **reinforcement_workers**: Number of reinforcement worker threads (default: `1`)
- **process_reaper_workers**: Number of process reaper worker threads (default: `1`)

#### Memory Configuration
- **main_segment_size**: Main memory segment size (default: `1G`)
- **client_data_segment_size**: Client data segment size (default: `512M`)
- **runtime_data_segment_size**: Runtime data segment size (default: `512M`)

#### Network Configuration
- **port**: ZeroMQ port for networking (default: `5555`)

#### Logging
- **log_level**: Logging level - `debug`, `info`, `warning`, `error` (default: `info`)

#### Performance Tuning
- **stack_size**: Stack size per task in bytes (default: `65536`)
- **queue_depth**: Task queue depth (default: `10000`)
- **heartbeat_interval**: Runtime heartbeat interval in milliseconds (default: `1000`)
- **task_timeout**: Task timeout in milliseconds (default: `30000`)

### Custom Pipeline

Create your own pipeline YAML:

```yaml
name: my_iowarp_deployment

pkgs:
  - pkg_type: jarvis_wrp_runtime.wrp_runtime
    pkg_name: runtime

    # Worker configuration
    low_latency_workers: 8
    high_latency_workers: 4

    # Memory configuration
    main_segment_size: "2G"
    client_data_segment_size: "1G"
    runtime_data_segment_size: "1G"

    # Logging
    log_level: "debug"
```

### Environment Variables

The package sets the following environment variable:

- **CHI_SERVER_CONF**: Path to generated runtime configuration file (in `shared_dir`)

This is the environment variable that both `RuntimeInit()` and `ClientInit()` check to load the configuration.

### Generated Configuration

The package generates a `chimaera_config.yaml` file in the pipeline's `shared_dir` with the format matching `config/chimaera_default.yaml`:

```yaml
# Worker thread configuration
workers:
  low_latency_threads: 4
  high_latency_threads: 2
  reinforcement_threads: 1
  process_reaper_threads: 1

# Memory segment configuration
memory:
  main_segment_size: 1073741824
  client_data_segment_size: 536870912
  runtime_data_segment_size: 536870912

# Network configuration
network:
  port: 5555

# ... (additional configuration)
```

## Package Architecture

The `wrp_runtime` package is a Service-type package that:

1. **Configuration Phase** (`_configure`):
   - Generates Chimaera runtime configuration in `shared_dir`
   - Sets `CHI_SERVER_CONF` environment variable
   - All configuration parameters match those in `config/chimaera_default.yaml`

2. **Start Phase** (`start`):
   - Verifies `chimaera_start_runtime` is available
   - Launches runtime on all nodes using `PsshExecInfo` with `env`
   - The runtime reads configuration from `CHI_SERVER_CONF`

3. **Stop Phase** (`stop`):
   - Uses `chimaera_stop_runtime` for graceful shutdown
   - Falls back to `kill()` if stop binary is unavailable

4. **Kill Phase** (`kill`):
   - Forcibly terminates runtime processes on all nodes

5. **Clean Phase** (`clean`):
   - Removes configuration and log files
   - Cleans shared memory segments (`/dev/shm/chi_*`) on all nodes

## Requirements

- IOWarp (Chimaera) runtime installed with binaries in `PATH`:
  - `chimaera_start_runtime`
  - `chimaera_stop_runtime`
- Jarvis-CD framework
- Parallel SSH (pssh) for distributed node management
- Hostfile configured in Jarvis for multi-node deployment

## Development Notes

The package follows Jarvis package development guidelines:

- **Class name**: `WrpRuntime` (UpperCamelCase from `wrp_runtime`)
- **Base class**: `Service` (from `jarvis_cd.core.pkg`)
- **Uses `env` for distributed execution** (not `mod_env`) as required
- **Configuration stored in `shared_dir`** for pipeline-wide access
- **Environment variable**: Sets `CHI_SERVER_CONF` (not `CHIMAERA_CONFIG`)
- **No file path requirements**: Assumes installed binaries are in `PATH`

## Pipeline Index

Example pipelines are available in the `pipelines/` directory:

- `basic_runtime.yaml`: Basic IOWarp runtime deployment with default settings

List available pipelines:
```bash
jarvis ppl index list jarvis_wrp_runtime
```

Load a pipeline:
```bash
jarvis ppl index load jarvis_wrp_runtime.basic_runtime
```

## License

This package is part of the IOWarp project.
