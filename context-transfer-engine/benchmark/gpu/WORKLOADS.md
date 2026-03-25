# GPU Workload Benchmark Algorithms

This document describes the conceptual algorithm for each workload benchmark,
how it maps to CTE and BaM execution models, and the data movement patterns
that drive the performance comparison.

---

## 1. PageRank — Iterative Graph Algorithm

### The Science

PageRank computes an importance score for every vertex in a directed graph.
The score of a vertex is the weighted sum of scores contributed by all vertices
that link to it. The algorithm iterates until convergence.

**Governing equation:**

```
PR(v) = (1 - α) + α × Σ PR(u) / deg(u)    for all u → v
```

Where `α = 0.85` (damping factor) and `deg(u)` is the out-degree of vertex `u`.

### Algorithm (Push-based)

```
Input:  CSR graph (offsets[], edges[]), α, tolerance
Output: PR scores per vertex

1. Initialize: value[v] = 1.0 for all v, residual[v] = 0

2. Repeat until convergence:
   a. PUSH phase (parallel over vertices):
      For each vertex v:
        contribution = α × value[v] / degree(v)
        For each neighbor u in edges[offsets[v] .. offsets[v+1]]:
          atomicAdd(residual[u], contribution)

   b. UPDATE phase (parallel over vertices):
      For each vertex v:
        If |residual[v]| > tolerance:
          value[v] += residual[v]
          mark as active
        residual[v] = 0

   c. If no vertex is active → converged, stop
```

### Data Structures

| Array | Type | Size | Access Pattern |
|-------|------|------|----------------|
| `offsets` | uint64 | V+1 | Sequential per vertex |
| `edges` | uint32 | E | Irregular (adjacency-driven) |
| `values` | float | V | Read per vertex, write in update |
| `residuals` | float | V | Atomic scatter (random writes) |

### GPU Parallelism

- **Warp-per-vertex**: each warp processes one vertex's edge list
- Lane 0 computes the contribution, all 32 lanes stride over edges
- `atomicAdd` to residuals handles concurrent writes from different warps

### Why This Stresses Data Movement

The edge list `edges[]` is the dominant data — for large graphs it can be
tens of GB. The access pattern is irregular: each vertex reads a different
contiguous range of edges, but the ranges vary in length (power-law degree
distribution). This makes caching effective (hot hub vertices are accessed
often) but sequential prefetching ineffective (access offsets are data-dependent).

### CTE vs BaM

- **CTE**: Loads edge chunks via `AsyncGetBlob` into HBM, computes on HBM.
  Bulk transfer amortizes I/O overhead. Each warp owns a vertex partition
  and its corresponding edge range.
- **BaM**: GPU threads call `edges.read(i)` which goes through the page cache.
  Each edge access triggers an atomicCAS tag check. Hot edges stay in HBM
  across iterations; cold edges are fetched from DRAM on demand.

### Tunable Parameters

| CLI Flag | Default | Effect on Data Size |
|----------|---------|---------------------|
| `--vertices V` | 100,000 | Number of vertices. Edge count ≈ V × avg_degree. |
| `--avg-degree D` | 16 | Average out-degree. Edge list = V×D×4 bytes. |
| `--iterations N` | 10 | PR iterations (convergence loops). |

**Data size formula:**

```
edge_bytes  = num_edges × 4          (num_edges ≈ V × D after dedup)
offset_bytes = (V + 1) × 8
total_HBM   ≈ edge_bytes             (offsets + values + residuals are small)
```

**Examples:**

| V | D | Edges (approx) | Edge List | Total Working Set |
|---|---|----------------|-----------|-------------------|
| 10K | 8 | ~71K | 0.3 MB | ~0.3 MB |
| 100K | 16 | ~1.4M | 5.4 MB | ~5.8 MB |
| 1M | 32 | ~28M | 107 MB | ~115 MB |
| 10M | 64 | ~560M | 2.1 GB | ~2.3 GB |

### Source of Graph

Synthetic R-MAT (Recursive Matrix) generator with parameters `a=0.57, b=0.19,
c=0.19, d=0.05`. Produces power-law degree distributions similar to real-world
social networks and web graphs.

---

## 2. GNN Feature Aggregation — Graph Neural Network

### The Science

Graph Neural Networks learn node representations by aggregating feature vectors
from a node's neighborhood. The simplest GNN layer (GraphSAGE mean aggregation)
computes a new feature vector for each node by averaging its own features with
those of its neighbors:

**Governing equation:**

```
h_v' = MEAN({h_v} ∪ {h_u : u ∈ N(v)})
     = (h_v + Σ h_u) / (1 + |N(v)|)
```

Where `h_v ∈ ℝ^d` is the d-dimensional feature vector of node v, and `N(v)` is
the set of neighbors of v in the graph.

### Algorithm

```
Input:  features[V × D], adjacency list, adjacency offsets
Output: output[V × D]

For each node v (parallel):
  nbr_start = adj_offsets[v]
  nbr_end   = adj_offsets[v + 1]

  For each feature dimension f in [0, D):
    sum = features[v × D + f]           // self-feature

    For each neighbor index j in [nbr_start, nbr_end):
      u = adj_list[j]
      sum += features[u × D + f]        // neighbor feature

    output[v × D + f] = sum / (1 + num_neighbors)  // mean aggregation
```

### Data Structures

| Array | Type | Size | Access Pattern |
|-------|------|------|----------------|
| `features` | float | V × D | Random (neighbor-driven lookups) |
| `adj_list` | uint32 | E | Sequential per node |
| `adj_offsets` | uint64 | V+1 | Sequential |
| `output` | float | V × D | Sequential write per node |

### GPU Parallelism

- **Thread-per-node**: each thread computes the aggregation for one node
- Inner loops over D features and E neighbors are sequential per thread
- Memory-bound: dominated by random feature reads for neighbors

### Why This Stresses Data Movement

The feature table can be enormous — for IGB-260M (260M nodes × 1024 floats),
it's ~1 TB. During a mini-batch, the GPU reads feature vectors for randomly
sampled nodes AND their neighbors. The neighbor lookups create a scattered
access pattern that defeats sequential prefetching. This is the exact workload
that GIDS (OSDI'24) targets: large feature stores that don't fit in GPU memory.

### CTE vs BaM

- **CTE**: Loads the full feature table via `AsyncGetBlob` into HBM, then
  computes all aggregations from HBM at full bandwidth. The table is loaded
  once per iteration and reused for all node computations.
- **BaM**: Each `features.read(u * D + f)` goes through the page cache. For
  random neighbor access, many pages are touched, and the hit rate depends on
  graph locality. Pages for high-degree hub nodes stay hot in cache.

### Tunable Parameters

| CLI Flag | Default | Effect on Data Size |
|----------|---------|---------------------|
| `--num-nodes V` | 500,000 | Number of nodes in the graph. |
| `--emb-dim D` | 128 | Feature embedding dimension per node. |
| `--avg-degree K` | 16 | Average neighbor count per node. |
| `--iterations N` | 10 | Number of aggregation iterations. |

**Data size formula:**

```
feature_bytes = V × D × 4            (float32 per dimension)
adj_bytes     = num_edges × 4        (num_edges ≈ V × K)
total_HBM     ≈ feature_bytes        (features dominate; adj in HBM always)
```

**Examples:**

| V | D | Feature Table | Adj List (K=16) | Total |
|---|---|---------------|-----------------|-------|
| 10K | 64 | 2.4 MB | 0.6 MB | 3 MB |
| 100K | 128 | 48.8 MB | 6.1 MB | 55 MB |
| 1M | 256 | 976 MB | 61 MB | 1.0 GB |
| 10M | 1024 | 38.1 GB | 610 MB | 38.7 GB |

The 10M × 1024 case approximates the IGB-260M dataset scale from GIDS.

### Source of Graph

Synthetic random graph with configurable average degree. Nodes have random
float features drawn from uniform [-1, 1].

---

## 3. LLM KV Cache — Large Language Model Inference

### The Science

During autoregressive LLM inference (text generation), each new token requires
attending over all previously generated tokens. The Key-Value (KV) cache stores
the key and value projections from all previous positions so they don't need to
be recomputed. For each new token, the model:

1. Computes Q (query) from the new token
2. Loads K (keys) and V (values) from the cache for all previous positions
3. Computes attention: `softmax(Q · K^T / √d) · V`
4. Appends the new token's K, V to the cache

**Attention equation (simplified to argmax for benchmarking):**

```
score(s) = (Q · K[s]) / √d_head     for each cached position s
best     = argmax_s score(s)
output   = V[best]
```

### Algorithm

```
Input:  KV cache [2 × num_heads × seq_len × head_dim] per layer
        Query Q [num_heads × head_dim]
Output: Attention output [num_heads × head_dim]

For each decode token t:
  For each transformer layer l:
    KV = load_kv_cache(layer=l)          // from storage

    For each attention head h (parallel):
      K = KV[0, h, :, :]                 // shape [seq_len, head_dim]
      V = KV[1, h, :, :]

      // Dot-product attention
      best_score = -∞
      best_pos = 0
      For each cached position s in [0, seq_len):
        dot = Σ Q[h, d] × K[s, d]  for d in [0, head_dim)  // warp-reduce
        score = dot / √head_dim
        If score > best_score:
          best_score = score
          best_pos = s

      output[h, :] = V[best_pos, :]      // copy best value vector

      // Append new KV entry
      K[t, :] = new_key[h, :]
      V[t, :] = new_value[h, :]

    save_kv_cache(layer=l, KV)            // to storage
```

### Data Structures

| Array | Type | Size | Access Pattern |
|-------|------|------|----------------|
| KV cache | float | 2 × H × S × D per layer | Sequential scan (K), point lookup (V) |
| Query | float | H × D | Read once per token |
| Output | float | H × D | Write once per token |

Where H = num_heads, S = seq_len, D = head_dim. Total KV per layer =
`2 × H × S × D × 4` bytes. For GPT-2 (12 layers, 12 heads, 64 dim, 2048 seq):
12 MB/layer, 144 MB total.

### GPU Parallelism

- **Warp-per-head**: each warp computes attention for one head
- All 32 lanes cooperate on the dot-product reduction (`__shfl_down_sync`)
- Sequential scan over `seq_len` positions (memory-bound for long sequences)

### Why This Stresses Data Movement

The KV cache grows linearly with sequence length and must be loaded for every
new token. For long contexts (32K+ tokens) with large models (70B+), the KV
cache is hundreds of GB — far exceeding GPU memory. The access pattern is
sequential within each head (scan over positions) but the total data volume
is enormous. This is the workload that GeminiFS (FAST'25) targets.

### Tunable Parameters

| CLI Flag | Default | Effect on Data Size |
|----------|---------|---------------------|
| `--num-layers L` | 12 | Transformer layers. KV scales linearly. |
| `--num-heads H` | 12 | Attention heads per layer. |
| `--head-dim D` | 64 | Dimension per head. |
| `--seq-len S` | 2,048 | Maximum sequence length (KV cache depth). |
| `--decode-tokens T` | 32 | Tokens to generate (decode iterations). |

**Data size formula:**

```
kv_per_layer  = 2 × H × S × D × 4   bytes (K + V, float32)
kv_total      = L × kv_per_layer
per_token_IO  = L × kv_per_layer     (read all layers) + L × 2×H×D×4 (write new entries)
```

**Examples (model configurations):**

| Model | L | H | D | S | KV/Layer | KV Total | Per-Token Read |
|-------|---|---|---|---|----------|----------|----------------|
| GPT-2 Small | 12 | 12 | 64 | 2K | 12 MB | 144 MB | 144 MB |
| LLaMA-7B | 32 | 32 | 128 | 4K | 128 MB | 4 GB | 4 GB |
| LLaMA-70B | 80 | 64 | 128 | 8K | 512 MB | 40 GB | 40 GB |

### CTE vs BaM

- **CTE**: Per-layer GetBlob/PutBlob cycle. Each warp loads its heads' KV via
  `AsyncGetBlob`, computes attention, writes back updated KV via `AsyncPutBlob`.
  The per-layer granularity allows pipelining: load layer L+1 while computing
  on layer L (not yet implemented — currently sequential).
- **BaM**: `kv_cache.read(offset)` for every K element during the attention scan.
  Each position-head pair generates `head_dim` page cache lookups. Sequential
  scan within a head has good spatial locality (adjacent elements in same page).

---

## 4. Gray-Scott — Reaction-Diffusion Stencil Simulation

### The Science

The Gray-Scott model simulates two chemical species (U and V) that react and
diffuse in a 3D domain. It produces complex spatiotemporal patterns (spots,
stripes, waves) depending on the feed rate F and kill rate k.

**Governing PDEs:**

```
∂u/∂t = Du·∇²u − u·v² + F·(1 − u)
∂v/∂t = Dv·∇²v + u·v² − (F + k)·v
```

Where:
- `u, v` are concentrations of the two species
- `Du = 0.05, Dv = 0.1` are diffusion coefficients
- `F = 0.04` is the feed rate (replenishes u)
- `k = 0.06075` is the kill rate (removes v)
- `∇²` is the 3D Laplacian operator

### Algorithm (Forward Euler with 7-point stencil)

```
Input:  u[L³], v[L³] — 3D concentration fields
        Parameters: Du, Dv, F, k, dt
Output: u'[L³], v'[L³] — updated fields

For each timestep:
  For each grid point (x, y, z) (parallel):
    // 7-point finite difference Laplacian
    lap_u = (u[x-1,y,z] + u[x+1,y,z]
           + u[x,y-1,z] + u[x,y+1,z]
           + u[x,y,z-1] + u[x,y,z+1]
           - 6 × u[x,y,z]) / 6

    lap_v = (same pattern for v)

    // Reaction terms
    uvv = u[x,y,z] × v[x,y,z]²

    // Forward Euler update
    u'[x,y,z] = u[x,y,z] + dt × (Du × lap_u − uvv + F × (1 − u[x,y,z]))
    v'[x,y,z] = v[x,y,z] + dt × (Dv × lap_v + uvv − (F + k) × v[x,y,z])

  Swap(u, u')
  Swap(v, v')
```

### Data Structures

| Array | Type | Size | Access Pattern |
|-------|------|------|----------------|
| u, v | float | L³ each | 7-point stencil (6 neighbors + center) |
| u', v' | float | L³ each | Dense write (one output per point) |

With periodic boundary conditions: `x-1` wraps to `L-1`, `x+1` wraps to `0`.

### GPU Parallelism

- **Thread-per-point**: each thread computes one grid point's update
- 7 reads from u + 7 reads from v = 14 reads per point (high arithmetic intensity)
- Dense, regular access pattern — excellent spatial locality

### Why This Stresses Data Movement

The two concentration fields must be read every timestep (14 reads per point).
For large grids (512³ = 134M points), each field is 512 MB — total working set
is 2 GB for reads plus 1 GB for writes per step. The stencil pattern has
excellent spatial locality (neighbors are adjacent in memory) so caching is
very effective. Periodic checkpointing adds write bursts.

### Tunable Parameters

| CLI Flag | Default | Effect on Data Size |
|----------|---------|---------------------|
| `--grid-size L` | 128 | Cubic grid dimension. Total points = L³. |
| `--steps N` | 100 | Simulation timesteps. |
| `--checkpoint-freq C` | 10 | Checkpoint (PutBlob) every C steps. 0 = no checkpoints. |

**Data size formula:**

```
field_bytes   = L³ × 4               (one float32 field)
working_set   = 4 × field_bytes      (u, v, u', v')
per_step_IO   = 2 × field_bytes      (read u,v) + 2 × field_bytes (write u',v')
checkpoint_IO = 2 × field_bytes      (write u,v snapshot every C steps)
```

**Examples:**

| L | Grid Points | Per Field | Working Set (4 fields) | Per-Step I/O |
|---|-------------|-----------|------------------------|--------------|
| 32 | 32K | 128 KB | 512 KB | 512 KB |
| 64 | 262K | 1 MB | 4 MB | 4 MB |
| 128 | 2.1M | 8 MB | 32 MB | 32 MB |
| 256 | 16.8M | 64 MB | 256 MB | 256 MB |
| 512 | 134M | 512 MB | 2 GB | 2 GB |

### CTE vs BaM

- **CTE**: Per-step GetBlob/PutBlob cycle. Each warp loads its grid slice via
  `AsyncGetBlob`, computes the stencil (all 32 lanes), writes back via
  `AsyncPutBlob`. The stencil has halo dependencies (neighbors at partition
  boundaries), so each warp loads the full field and computes only its slice.
- **BaM**: `u_arr.read(gs_idx(x,y,z,L))` for each stencil neighbor. The 7-point
  stencil has excellent spatial locality — most neighbors fall on the same page.
  The per-element atomicCAS overhead is the bottleneck, not cache misses.

---

## Comparison Summary

| Workload | Dominant Operation | Access Pattern | Data Size Driver |
|----------|-------------------|----------------|------------------|
| PageRank | Edge traversal | Irregular (power-law) | Edge count E |
| GNN | Feature gather | Random (neighbor-driven) | Nodes × embedding dim |
| LLM KV | Attention scan | Sequential per head | Layers × heads × seq_len × dim |
| Gray-Scott | Stencil compute | Regular (7-point) | Grid volume L³ |

### When CTE Wins

CTE excels when data can be loaded in bulk and computed upon at full HBM
bandwidth. The `AsyncGetBlob` overhead is amortized over large transfers.
CTE's advantage grows with data size because the per-blob overhead becomes
negligible relative to the compute time.

### When BaM Wins

BaM is better suited for fine-grained, unpredictable access patterns where
only a subset of the data is needed per iteration and caching provides benefit
across iterations. However, the current per-element `atomicCAS` overhead in
the page cache makes BaM slower than CTE's bulk transfer for all tested
workloads. BaM's advantage would appear with NVMe storage (not DRAM), where
the page cache avoids expensive SSD round-trips for cached data.
