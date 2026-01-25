# IoWarp Runtime Deployment Guide

This guide describes how to deploy and configure the IoWarp runtime (Chimaera distributed task execution framework).

## Table of Contents

- [Quick Start](#quick-start)
- [Configuration Methods](#configuration-methods)
- [Environment Variables](#environment-variables)
- [Configuration File Format](#configuration-file-format)
  - [Complete Configuration Example](#complete-configuration-example)
  - [Configuration Parameters Reference](#configuration-parameters-reference)
  - [Compose Configuration](#compose-configuration)
- [Deployment Scenarios](#deployment-scenarios)
- [Troubleshooting](#troubleshooting)
- [Configuration Best Practices](#configuration-best-practices)

## Quick Start

### Basic Deployment

```bash
# Set configuration file path
export CHI_SERVER_CONF=/path/to/chimaera_config.yaml

# Start the runtime
chimaera_start_runtime
```

### Docker Deployment

```bash
cd docker
docker-compose up -d
```

## Configuration Methods

The runtime supports multiple configuration methods with the following precedence:

1. **Environment Variable (Recommended)**: `CHI_SERVER_CONF` or `WRP_RUNTIME_CONF`
   - Points to a YAML configuration file
   - Most flexible and explicit method
   - `CHI_SERVER_CONF` is checked first, then `WRP_RUNTIME_CONF`

2. **Default Configuration**: Built-in defaults
   - Used when no configuration file is specified
   - Suitable for development and testing

### Configuration File Path Resolution

The runtime reads the configuration file path from environment variables with the following precedence:

1. **CHI_SERVER_CONF** (checked first)
2. **WRP_RUNTIME_CONF** (fallback if CHI_SERVER_CONF is not set)
3. Built-in defaults (if neither environment variable is set)

**Examples**:

```bash
# Method 1: Using CHI_SERVER_CONF (recommended)
export CHI_SERVER_CONF=/etc/chimaera/chimaera_config.yaml
chimaera_start_runtime

# Method 2: Using WRP_RUNTIME_CONF (alternative)
export WRP_RUNTIME_CONF=/etc/iowarp/runtime_config.yaml
chimaera_start_runtime

# Method 3: No configuration (uses defaults)
chimaera_start_runtime
```

## Environment Variables

### Configuration File Location

| Variable | Description | Default | Priority |
|----------|-------------|---------|----------|
| `CHI_SERVER_CONF` | Path to YAML configuration file | (empty - uses defaults) | Primary |
| `WRP_RUNTIME_CONF` | Alternative path to YAML configuration file | (empty - uses defaults) | Secondary |

**Note**: The runtime checks `CHI_SERVER_CONF` first. If not set, it falls back to `WRP_RUNTIME_CONF`. If neither is set, built-in defaults are used.

**Important**: The runtime does NOT read individual `CHI_*` environment variables (like `CHI_SCHED_WORKERS`, `CHI_ZMQ_PORT`, etc.). All configuration must be specified in a YAML file pointed to by `CHI_SERVER_CONF` or `WRP_RUNTIME_CONF`.

## Configuration File Format

The configuration file uses YAML format with the following sections:

### Complete Configuration Example

```yaml
# Chimaera Runtime Configuration
# Based on config/chimaera_default.yaml

# Worker thread configuration
workers:
  sched_threads: 4           # Scheduler worker threads (for fast tasks with EstCpuTime < 50us)
  slow_threads: 4            # Slow worker threads (for long-running tasks with EstCpuTime >= 50us)

# Memory segment configuration
memory:
  main_segment_size: 1073741824      # 1GB (or use: 1G)
  client_data_segment_size: 536870912 # 512MB (or use: 512M)
  runtime_data_segment_size: 536870912 # 512MB (or use: 512M)

# Network configuration
networking:
  port: 5555
  neighborhood_size: 32  # Maximum number of queries when splitting range queries
  hostfile: "/etc/chimaera/hostfile"  # Optional: path to hostfile for distributed mode
  wait_for_restart: 30   # Seconds to wait for remote connection during system boot
  wait_for_restart_poll_period: 1  # Seconds between retry attempts

# Logging configuration
logging:
  level: "info"  # Options: debug, info, warning, error
  file: "/tmp/chimaera.log"

# Runtime configuration
runtime:
  stack_size: 65536  # 64KB per task
  queue_depth: 10000
  local_sched: "default"  # Local task scheduler (default: "default")
  heartbeat_interval: 1000  # milliseconds
```

### Configuration Parameters Reference

#### Worker Threads (`workers` section)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sched_threads` | integer | 4 | Number of scheduler worker threads for fast tasks (EstCpuTime < 50us) |
| `slow_threads` | integer | 4 | Number of slow worker threads for long-running tasks (EstCpuTime >= 50us) |

**Notes:**
- Fast tasks are routed to scheduler threads for low-latency execution
- Slow tasks are routed to dedicated slow threads to avoid blocking fast tasks
- Set based on CPU core count and workload characteristics
- Total threads = `sched_threads + slow_threads`

#### Memory Segments (`memory` section)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `main_segment_size` | size | 1GB | Main shared memory segment for task metadata and control structures |
| `client_data_segment_size` | size | 512MB | Client-side data segment for application data |
| `runtime_data_segment_size` | size | 512MB | Runtime-side data segment for internal state |

**Size format:** Supports bytes (`1073741824`) or suffixed values (`1G`, `512M`, `64K`)

**Docker requirements:** Set `shm_size` >= sum of all segments (recommend 20-30% extra)

#### Network Configuration (`networking` section)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `port` | integer | 5555 | ZeroMQ port for distributed communication |
| `neighborhood_size` | integer | 32 | Maximum number of nodes queried when splitting range queries |
| `hostfile` | string | (none) | Path to hostfile containing cluster node IP addresses (one per line) |
| `wait_for_restart` | integer | 30 | Seconds to wait for remote connection during system boot |
| `wait_for_restart_poll_period` | integer | 1 | Seconds between connection retry attempts |

**Notes:**
- Port must match across all cluster nodes
- Larger `neighborhood_size` improves load distribution but increases network overhead
- Smaller values (4-8) useful for stress testing
- `hostfile` required for distributed deployments
- `wait_for_restart` prevents failures when remote nodes are still booting
- `wait_for_restart_poll_period` controls retry frequency (lower = more frequent retries)

#### Logging Configuration (`logging` section)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | string | "info" | Log level: `debug`, `info`, `warning`, `error` |
| `file` | string | "/tmp/chimaera.log" | Path to log file |

**Log levels:**
- `debug`: Detailed debugging information (development only)
- `info`: General operational information (recommended for testing)
- `warning`: Warning messages only (production)
- `error`: Error messages only (production)

#### Runtime Configuration (`runtime` section)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stack_size` | integer | 65536 | Stack size per task in bytes (64KB default) |
| `queue_depth` | integer | 10000 | Maximum depth of task queues |
| `local_sched` | string | "default" | Local task scheduler (default: "default") |
| `heartbeat_interval` | integer | 1000 | Heartbeat interval in milliseconds |

**Local scheduler options:**
- `default`: Default task scheduler with factory-based task dispatching

#### Connection Retry During System Boot

When deploying distributed clusters, nodes may not all become available simultaneously. The `wait_for_restart` feature provides automatic retry logic for remote connections during system boot:

**How it works:**
1. When SendIn attempts to send a task to a remote node and the connection fails
2. The system waits `wait_for_restart_poll_period` seconds and retries
3. This continues until either:
   - The connection succeeds, OR
   - `wait_for_restart` seconds have elapsed (timeout)
4. During the wait period, the task yields control using `task->Wait()` to avoid blocking the worker

**Configuration parameters:**
- `wait_for_restart`: Maximum time to wait for connection (default: 30 seconds)
- `wait_for_restart_poll_period`: Time between retry attempts (default: 1 second)

**Example scenarios:**
```yaml
# Quick timeout for fast-starting systems
networking:
  wait_for_restart: 10
  wait_for_restart_poll_period: 1

# Extended timeout for slow-starting systems
networking:
  wait_for_restart: 60
  wait_for_restart_poll_period: 2

# Frequent retries for flaky networks
networking:
  wait_for_restart: 30
  wait_for_restart_poll_period: 0.5
```

**Use cases:**
- **Container orchestration**: Nodes starting at different times in Docker/Kubernetes
- **VM deployments**: VMs with different boot times
- **Network delays**: Temporary network partitions during startup
- **Rolling restarts**: Nodes restarting in sequence

**Best practices:**
- Set `wait_for_restart` based on expected maximum boot time difference
- Use shorter `wait_for_restart_poll_period` for more responsive retries
- Monitor logs for "retrying" messages to tune timeout values
- In production, set `wait_for_restart` to 2-3x typical boot time variance

### Size Format

Memory sizes can be specified in multiple formats:
- **Bytes**: `1073741824`
- **Suffixed**: `1G`, `512M`, `64K`
- **Human-readable**: Automatically parsed by HSHM ConfigParse

### Hostfile Format

For distributed deployments, create a hostfile with one IP address per line:

```
172.20.0.10
172.20.0.11
172.20.0.12
```

Then reference it in the configuration:

```yaml
networking:
  hostfile: "/etc/chimaera/hostfile"
```

### Compose Configuration

The `compose` section allows you to declaratively define pools that should be created when the runtime starts. This is useful for:
- Automated pool creation during deployment
- Infrastructure-as-code for distributed systems
- Testing and development environments

**Basic Compose Example:**

```yaml
# Chimaera configuration with compose section
workers:
  sched_threads: 4
  slow_threads: 4

memory:
  main_segment_size: 1GB
  client_data_segment_size: 512MB
  runtime_data_segment_size: 512MB

networking:
  port: 5555

compose:
  # BDev file-based storage device
  - mod_name: chimaera_bdev
    pool_name: /tmp/storage_device.dat
    pool_query: dynamic
    pool_id: 300.0
    capacity: 1GB
    bdev_type: file
    io_depth: 32
    alignment: 4096

  # BDev RAM-based storage device
  - mod_name: chimaera_bdev
    pool_name: ram_cache
    pool_query: local
    pool_id: 301.0
    capacity: 512MB
    bdev_type: ram
    io_depth: 64
    alignment: 4096

  # Custom ChiMod pool
  - mod_name: chimaera_custom_mod
    pool_name: my_custom_pool
    pool_query: dynamic
    pool_id: 400.0
    # ChiMod-specific parameters here
    custom_param1: value1
    custom_param2: value2
```

#### Compose Section Parameters

**Common Parameters (all pools):**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `mod_name` | string | Yes | ChiMod library name (e.g., "chimaera_bdev", "chimaera_admin") |
| `pool_name` | string | Yes | Pool name or identifier; for file-based BDev, this is the file path |
| `pool_query` | string | Yes | Pool routing: "dynamic" (recommended) or "local" |
| `pool_id` | string | Yes | Pool ID in format "major.minor" (e.g., "300.0") |

**BDev-Specific Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `capacity` | size | Yes | Total capacity of the block device (e.g., "1GB", "512MB") |
| `bdev_type` | string | Yes | Device type: "file" or "ram" |
| `io_depth` | integer | No | I/O queue depth (default: 16) |
| `alignment` | integer | No | Block alignment in bytes (default: 4096) |

**Pool Query Values:**
- `dynamic` (recommended): Automatically routes to local if pool exists, broadcast if creating new
- `local`: Create/access pool only on local node
- `broadcast`: Create pool on all nodes in cluster

**Pool ID Format:**
- Format: `"major.minor"` where major and minor are integers
- Example: `"300.0"`, `"301.5"`, `"1000.100"`
- Must be unique across all pools in the system

#### Using Compose with chimaera_compose Utility

The `chimaera_compose` utility creates pools from a compose configuration file. This is useful for:
- Setting up pools after runtime initialization
- Scripted deployment workflows
- Testing pool configurations

**Usage:**

```bash
# Start runtime first
export CHI_SERVER_CONF=/path/to/config.yaml
chimaera_start_runtime &

# Wait for runtime to initialize
sleep 2

# Create pools from compose configuration
chimaera_compose /path/to/config.yaml
```

#### Compose Best Practices

1. **Pool IDs**: Use a consistent numbering scheme (e.g., 300-399 for BDev, 400-499 for custom modules)
2. **Pool Names**: For file-based BDev, use absolute paths; for RAM-based BDev, use descriptive names
3. **Pool Query**: Prefer `dynamic` for automatic routing optimization
4. **Capacity**: Ensure capacity doesn't exceed available storage/memory
5. **Error Handling**: Always verify pool creation succeeded (check return codes)

#### Complete Compose Example

```yaml
# Production-ready configuration with multiple pools
workers:
  sched_threads: 8
  slow_threads: 8

memory:
  main_segment_size: 4GB
  client_data_segment_size: 2GB
  runtime_data_segment_size: 2GB

networking:
  port: 5555
  neighborhood_size: 32
  hostfile: "/etc/chimaera/hostfile"

logging:
  level: "info"
  file: "/var/log/chimaera/chimaera.log"

runtime:
  stack_size: 65536
  queue_depth: 20000
  local_sched: "default"
  heartbeat_interval: 1000

compose:
  # Primary storage device (file-based)
  - mod_name: chimaera_bdev
    pool_name: /data/primary_storage.dat
    pool_query: dynamic
    pool_id: 300.0
    capacity: 100GB
    bdev_type: file
    io_depth: 64
    alignment: 4096

  # Fast cache device (RAM-based)
  - mod_name: chimaera_bdev
    pool_name: fast_cache
    pool_query: local
    pool_id: 301.0
    capacity: 8GB
    bdev_type: ram
    io_depth: 128
    alignment: 4096

  # Secondary storage (file-based)
  - mod_name: chimaera_bdev
    pool_name: /data/secondary_storage.dat
    pool_query: dynamic
    pool_id: 302.0
    capacity: 500GB
    bdev_type: file
    io_depth: 32
    alignment: 4096
```

## Troubleshooting

### Issue: Configuration not loaded

**Symptoms**: Runtime uses default values instead of configuration file

**Solutions**:
1. Ensure `CHI_SERVER_CONF` or `WRP_RUNTIME_CONF` is set before starting runtime:
   ```bash
   echo $CHI_SERVER_CONF
   echo $WRP_RUNTIME_CONF
   ```
2. Check file permissions (must be readable):
   ```bash
   ls -l $CHI_SERVER_CONF
   ```
3. Verify file path is absolute, not relative
4. Check runtime logs for configuration loading messages

### Issue: Docker container shared memory exhausted

**Symptoms**: `Failed to allocate shared memory segment`

**Solutions**:
1. Increase Docker `shm_size`:
   ```yaml
   shm_size: 4gb  # Must be >= sum(main + client_data + runtime_data)
   ```

2. Reduce segment sizes in configuration:
   ```yaml
   memory:
     main_segment_size: 512M
     client_data_segment_size: 256M
     runtime_data_segment_size: 256M
   ```

### Issue: Network connection failures in distributed mode

**Symptoms**: Tasks not routing to remote nodes

**Solutions**:
1. Verify hostfile contains correct IP addresses:
   ```bash
   cat /etc/chimaera/hostfile
   ```

2. Check network connectivity:
   ```bash
   # Test connectivity to each node
   nc -zv 172.20.0.10 5555
   nc -zv 172.20.0.11 5555
   ```

3. Verify port configuration matches across nodes:
   ```yaml
   networking:
     port: 5555  # Must be same on all nodes
   ```

### Issue: High memory usage

**Symptoms**: Runtime consuming more memory than expected

**Solutions**:
1. Reduce segment sizes:
   ```yaml
   memory:
     main_segment_size: 512M
     client_data_segment_size: 256M
     runtime_data_segment_size: 256M
   ```

2. Reduce queue depth:
   ```yaml
   performance:
     queue_depth: 5000  # Lower value
   ```

3. Monitor with logging:
   ```yaml
   logging:
     level: "debug"  # Enable detailed logging
   ```

## Configuration Best Practices

1. **Configuration File Management**:
   - Always use YAML configuration files instead of relying on defaults
   - Keep configuration files in version control
   - Use descriptive names for configuration files (e.g., `production.yaml`, `development.yaml`)
   - Document any deviations from default values with comments

2. **Memory Sizing**:
   - Set `main_segment_size` based on total task count and data size
   - Allocate at least 50% of main_segment_size for client/runtime segments
   - Ensure Docker `shm_size` is 20-30% larger than sum of segments
   - Example: If total segments = 2GB, set `shm_size: 2.5gb`

3. **Worker Threads**:
   - Set `sched_threads` + `slow_threads` equal to CPU core count
   - Balance between fast and slow threads based on workload (50/50 split is a good starting point)
   - Use more `sched_threads` for workloads with many fast tasks (< 50us execution time)
   - Use more `slow_threads` for workloads with many long-running tasks (>= 50us execution time)

4. **Network Tuning**:
   - Use smaller `neighborhood_size` (4-8) for stress testing
   - Use larger values (32-64) for production distributed deployments
   - Keep port consistent across all cluster nodes
   - Always specify hostfile path for distributed deployments

5. **Logging**:
   - Use `debug` level during development
   - Use `info` level for normal operation
   - Use `warning` or `error` for production
   - Ensure log directory is writable

6. **Runtime Configuration**:
   - Start with default `stack_size` (64KB) and increase if tasks need more
   - Increase `queue_depth` for bursty workloads
   - Use `round_robin` lane mapping for general workloads
   - Adjust `heartbeat_interval` based on monitoring requirements

## References

### Configuration Files
- **Default configuration**: `config/chimaera_default.yaml`
  - Reference implementation with all default values
  - Includes comments explaining each parameter
  - 4 scheduler workers, 4 slow workers
  - 1GB main segment, 512MB client/runtime segments

### Compose Utility
- **Compose utility source**: `util/chimaera_compose.cc`
  - Standalone tool for creating pools from compose configurations
  - Requires runtime to be initialized first
  - Usage: `chimaera_compose <config.yaml>`

- **Compose test script**: `test/unit/test_chimaera_compose.sh`
  - Complete example of using chimaera_compose utility
  - Demonstrates BDev pool creation from compose section
  - Includes verification and cleanup steps

### Source Code
- **Runtime startup**: `util/chimaera_start_runtime.cc`
  - Main runtime initialization and server startup
  - Loads configuration from CHI_SERVER_CONF or WRP_RUNTIME_CONF

- **Configuration manager**: `src/config_manager.cc`, `include/chimaera/config_manager.h`
  - YAML parsing and configuration structures
  - PoolConfig and ComposeConfig definitions
  - Environment variable resolution

### Docker Deployment
- **Dockerfile**: `docker/deploy.Dockerfile`
  - Container image definition with all dependencies

- **Docker Compose**: `docker/docker-compose.yml`
  - Multi-node cluster orchestration
  - Static IP assignment for predictable routing

- **Entrypoint script**: `docker/entrypoint.sh`
  - Runtime configuration generation
  - Environment variable substitution

### Related Documentation
- **Module Development Guide**: `docs/MODULE_DEVELOPMENT_GUIDE.md`
  - ChiMod development and integration
  - Compose integration for custom modules

- **Docker README**: `docker/README.md`
  - Comprehensive Docker deployment guide
  - Network configuration and troubleshooting
