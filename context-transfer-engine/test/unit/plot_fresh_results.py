#!/usr/bin/env python3
"""Grade and plot the two fresh-Claude arms (15-query sweep).

Inputs:
  fresh_results_arm1.json -- accuracy/answer paths, arm 1 (no Acropolis)
  fresh_results_arm2.json -- accuracy per (backend, level), arm 2
  arm1_tokens.json        -- real billed tokens, arm 1 (mined from JSONL)
  arm2_tokens.json        -- real billed tokens, arm 2 (per-query + per-config)

Outputs:
  fresh_fig1_accuracy_heat.png      -- per-config accuracy heatmap
  fresh_fig2_per_query_winners.png  -- per-query x per-config success matrix
  fresh_fig3_summary.png            -- accuracy + tool-calls + token cost
  fresh_fig4_pareto.png             -- accuracy vs token cost (per config)
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)

EXPECTED = {
    1:  "kg_backend_qdrant.h",
    2:  "kg_backend_elasticsearch.h",
    3:  "kg_backend_neo4j.h",
    4:  "hdf5_summary.h",
    5:  "set_depth.cc",
    6:  "test_gpu_submission_gpu.cc",
    7:  "ggml_iowarp_backend.cc",
    8:  "kvcache_manager",
    9:  "run_e2e_gpu_test.sh",
    10: "bench_repo_scan.cc",
    11: "kg_backend_bm25.h",
    12: "kg_backend_elasticsearch.h",
    13: "depth_controller.h",
    14: "embedding_client.h",
    15: "test_indexing_depth_config.cc",
}

QUERY_TEXTS = {
    1:  "Qdrant vector backend",
    2:  "Elasticsearch text backend",
    3:  "Neo4j knowledge graph backend",
    4:  "HDF5 metadata extractor",
    5:  "CLI: set indexing depth",
    6:  "Unit test: GPU submission (real GPU)",
    7:  "GGML weights stream via GpuVMM",
    8:  "KV-cache manager (llama.cpp)",
    9:  "E2E: KV cache restore on GPU",
    10: "Benchmark: LLM agent loop",
    11: "BM25 with distributed IDF",
    12: "Hybrid retrieval (RRF)",
    13: "Depth controller (xattr inheritance)",
    14: "OpenAI-compatible embeddings client",
    15: "Unit test: indexing-depth config",
}

BACKENDS = ["bm25", "elasticsearch-kw", "elasticsearch-vec",
            "elasticsearch-rrf", "qdrant", "neo4j-kw", "neo4j-rrf"]
LEVELS = [0, 1, 2]
N_QUERIES = 15
N_QUERIES_MAX = 15  # vmax for accuracy colorbar


def hit(answer_path, expected_substring):
    return answer_path is not None and expected_substring in answer_path


def grade():
    with open(os.path.join(HERE, "fresh_results_arm1.json")) as f:
        arm1 = json.load(f)
    with open(os.path.join(HERE, "fresh_results_arm2.json")) as f:
        arm2 = json.load(f)

    base = {}
    for r in arm1["results"]:
        base[r["id"]] = {
            "n_calls": r["n_calls"],
            "hit": hit(r["answer_path"], EXPECTED[r["id"]]),
        }

    cell = {}
    for b in BACKENDS:
        for lv in LEVELS:
            cell[(b, lv)] = [False] * N_QUERIES
    for r in arm2["results"]:
        qid = r["id"]
        for c in r["configs"]:
            cell[(c["backend"], c["level"])][qid - 1] = hit(
                c["answer_path"], EXPECTED[qid])
    return base, cell


def load_tokens():
    """Return (arm1_per_query, arm2_per_query, arm2_per_config_total).

    arm1_per_query[qid]      -> billable_equiv int
    arm2_per_query[qid]      -> billable_equiv int  (cost across all 21 configs)
    arm2_per_config_total[(b,lv)] -> billable_equiv int summed over 15 queries
                                     (i.e. total cost if you used ONLY this
                                     config for all 15 queries)
    """
    with open(os.path.join(HERE, "arm1_tokens.json")) as f:
        a1 = json.load(f)
    with open(os.path.join(HERE, "arm2_tokens.json")) as f:
        a2 = json.load(f)

    arm1_pq = {r["id"]: r["billable_equiv"] for r in a1["per_query"]}
    arm2_pq = {r["id"]: r["billable_equiv"] for r in a2["per_query"]}

    arm2_pc_total = {(b, lv): 0 for b in BACKENDS for lv in LEVELS}
    arm2_pc_perq = {(b, lv): {} for b in BACKENDS for lv in LEVELS}
    for r in a2["per_config"]:
        if r["id"] is None:
            continue
        key = (r["backend"], r["level"])
        if key in arm2_pc_total:
            arm2_pc_total[key] += r["billable_equiv"]
            arm2_pc_perq[key][r["id"]] = r["billable_equiv"]

    return arm1_pq, arm2_pq, arm2_pc_total, arm2_pc_perq


def fig1_heat(base, cell):
    mat = np.zeros((len(BACKENDS), len(LEVELS)))
    for bi, b in enumerate(BACKENDS):
        for li, lv in enumerate(LEVELS):
            mat[bi, li] = sum(cell[(b, lv)])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, cmap="YlGn", vmin=0, vmax=N_QUERIES_MAX, aspect="auto")
    for bi in range(len(BACKENDS)):
        for li in range(len(LEVELS)):
            v = int(mat[bi, li])
            ax.text(li, bi, f"{v}/{N_QUERIES}", ha="center", va="center",
                    fontsize=12,
                    color="white" if v >= 0.7 * N_QUERIES_MAX else "black",
                    weight="bold")
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels([f"L{lv}" for lv in LEVELS], fontsize=11)
    ax.set_yticks(range(len(BACKENDS)))
    ax.set_yticklabels(BACKENDS, fontsize=10)

    base_hits = sum(1 for v in base.values() if v["hit"])
    base_calls = sum(v["n_calls"] for v in base.values())
    ax.set_title(
        f"Fresh Claude — Acropolis accuracy ({N_QUERIES} queries x backend x depth)\n"
        f"baseline (no Acropolis, Glob/Grep/Read): {base_hits}/{N_QUERIES} hits "
        f"in {base_calls} tool calls",
        fontsize=11)
    plt.colorbar(im, ax=ax,
                 label=f"queries answered correctly (0-{N_QUERIES})")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fresh_fig1_accuracy_heat.png"),
                dpi=140, bbox_inches="tight")
    print("wrote fresh_fig1_accuracy_heat.png")


def fig2_per_query(base, cell):
    fig, ax = plt.subplots(figsize=(15, 8))
    n_q = N_QUERIES
    n_cfg = len(BACKENDS) * len(LEVELS) + 1   # +1 for baseline column
    cfgs = ["baseline"] + [f"{b}-L{lv}" for b in BACKENDS for lv in LEVELS]

    grid = np.zeros((n_q, n_cfg))
    for qi in range(n_q):
        grid[qi, 0] = 1 if base[qi + 1]["hit"] else 0
        for ci, (b, lv) in enumerate(
                [(b, lv) for b in BACKENDS for lv in LEVELS]):
            grid[qi, ci + 1] = 1 if cell[(b, lv)][qi] else 0

    cmap = plt.cm.colors.ListedColormap(["#fbb", "#9c8"])
    im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    for qi in range(n_q):
        for ci in range(n_cfg):
            mark = "Y" if grid[qi, ci] else "."
            ax.text(ci, qi, mark, ha="center", va="center",
                    fontsize=9,
                    color="darkgreen" if grid[qi, ci] else "darkred",
                    weight="bold")
    ax.set_xticks(range(n_cfg))
    ax.set_xticklabels(cfgs, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(n_q))
    ax.set_yticklabels([f"q{qi+1}: {QUERY_TEXTS[qi+1]}" for qi in range(n_q)],
                      fontsize=8.5)
    ax.axvline(0.5, color="black", linewidth=1.5)   # baseline boundary
    ax.set_title("Per-query x per-config success matrix\n"
                 "(Y = correct file identified from sem_q result alone)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fresh_fig2_per_query_winners.png"),
                dpi=140, bbox_inches="tight")
    print("wrote fresh_fig2_per_query_winners.png")


def fig3_summary(base, cell, arm1_pq, arm2_pc_total):
    base_hits = sum(1 for v in base.values() if v["hit"])
    base_calls = sum(v["n_calls"] for v in base.values())

    by_lv = {lv: 0 for lv in LEVELS}
    for lv in LEVELS:
        for b in BACKENDS:
            by_lv[lv] += sum(cell[(b, lv)])
    by_lv_avg = {lv: by_lv[lv] / len(BACKENDS) for lv in LEVELS}

    # mean tokens/query for arm 1 (sum of per-query / 15)
    base_mean_tok = sum(arm1_pq.values()) / N_QUERIES
    # Depth-differentiated estimate (cache-propagation model from fig6).
    # Uniform per-config attribution can't separate L0/L1/L2 from raw data,
    # so we use the modeled estimate: total / 15 = per-query mean.
    est_total_per_lv = estimate_session_tokens_per_level(arm2_pc_total)
    lv_mean_tok = {lv: est_total_per_lv[lv] / N_QUERIES for lv in LEVELS}
    # Best SINGLE FIXED config: highest hits, then lowest tokens.
    config_score = []
    for b in BACKENDS:
        for lv in LEVELS:
            hits = sum(cell[(b, lv)])
            tok_per_q = arm2_pc_total[(b, lv)] / N_QUERIES
            config_score.append((hits, -tok_per_q, b, lv))
    config_score.sort(reverse=True)
    best_b, best_lv = config_score[0][2], config_score[0][3]
    best_hit = config_score[0][0]
    # Best config's per-query token cost = depth-modeled value for its level
    best_tok = lv_mean_tok[best_lv]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"Fresh Claude — baseline vs Acropolis ({N_QUERIES} queries)",
        fontsize=12)

    # Panel A: accuracy
    cfgs = ["baseline\n(Glob/Grep/Read)",
            "Acro L0\n(avg backends)",
            "Acro L1\n(avg backends)",
            "Acro L2\n(avg backends)",
            f"Acro BEST\n({best_b}-L{best_lv})"]
    vals = [base_hits, by_lv_avg[0], by_lv_avg[1], by_lv_avg[2], best_hit]
    colors = ["#888", "#fbb479", "#7ed996", "#9c8aff", "#3a9d4a"]
    bars = axes[0].bar(cfgs, vals, color=colors, edgecolor="black",
                       linewidth=0.4)
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.2,
                     f"{v:.1f}/{N_QUERIES}", ha="center", va="bottom",
                     fontsize=9)
    axes[0].set_ylabel(f"queries correct (of {N_QUERIES})")
    axes[0].set_ylim(0, N_QUERIES + 2)
    axes[0].set_title("Accuracy")
    axes[0].grid(axis="y", linestyle=":", alpha=0.4)
    axes[0].tick_params(axis="x", labelsize=8)

    # Panel B: tool-call cost
    cfgs_b = ["baseline", "Acro\n(1 sem_q/query)"]
    call_vals = [base_calls, N_QUERIES]
    bars = axes[1].bar(cfgs_b, call_vals, color=["#888", "#9c8aff"],
                       edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, call_vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.4,
                     str(v), ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel(f"total tool calls ({N_QUERIES} queries)")
    axes[1].set_title("Tool-call cost")
    axes[1].grid(axis="y", linestyle=":", alpha=0.4)

    # Panel C: token cost (mean tokens per query, billable-equivalent)
    tok_cfgs = ["baseline\n(Glob/Grep/Read)",
                "Acro L0\n(avg backends)",
                "Acro L1\n(avg backends)",
                "Acro L2\n(avg backends)",
                f"Acro BEST\n({best_b}-L{best_lv})"]
    tok_vals = [base_mean_tok, lv_mean_tok[0], lv_mean_tok[1],
                lv_mean_tok[2], best_tok]
    bars = axes[2].bar(tok_cfgs, tok_vals, color=colors,
                       edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, tok_vals):
        axes[2].text(b.get_x() + b.get_width() / 2, v * 1.02,
                     f"{v / 1000:.1f}K", ha="center", va="bottom",
                     fontsize=9)
    axes[2].set_ylabel("mean tokens/query (billable-equiv)")
    axes[2].set_title("Token cost")
    axes[2].grid(axis="y", linestyle=":", alpha=0.4)
    axes[2].tick_params(axis="x", labelsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(HERE, "fresh_fig3_summary.png"),
                dpi=140, bbox_inches="tight")
    print("wrote fresh_fig3_summary.png")


def estimate_session_tokens_per_level(arm2_pc_total):
    """Cache-propagation estimate of total session tokens by level.

    Each MCP response stays in conversation cache for all subsequent turns.
    Total cache_read contribution from results across N queries:
      sum_{k=1..N} (k-1) * R_lv = N(N-1)/2 * R_lv

    Billable-equivalent (cache_read price ~10% of input price):
      cache_read_be(lv) = N(N-1)/2 * R_lv * 0.1

    Constant overhead per call comes from arm 2's per-call attribution
    (mean across all configs), minus the average payload contribution.
    """
    payload_chars = {0: 430, 1: 430, 2: 2540}
    payload_tok = {lv: c / 4 for lv, c in payload_chars.items()}

    mean_per_call = sum(arm2_pc_total.values()) / (len(arm2_pc_total) * N_QUERIES)
    mean_payload_tok = sum(payload_tok.values()) / 3
    overhead_per_call = mean_per_call - mean_payload_tok

    cache_factor = N_QUERIES * (N_QUERIES - 1) / 2 * 0.1

    est_total = {}
    for lv in LEVELS:
        per_call_total = N_QUERIES * (overhead_per_call + payload_tok[lv])
        cache_propagation = cache_factor * payload_tok[lv]
        est_total[lv] = per_call_total + cache_propagation
    return est_total


def fig6_depth_cost_estimate(arm1_pq, arm2_pc_total):
    """Estimated cumulative session cost by depth level (15 queries).

    Cost model: constant_overhead * 15 + 15 * payload(lv) + cache_propagation(lv)
    where payload sizes are measured from arm 2 transcript and the constant
    overhead is derived from arm 2's per-call attribution mean.
    """
    est_total = estimate_session_tokens_per_level(arm2_pc_total)
    base_total = sum(arm1_pq.values())

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.suptitle(
        "Estimated session token cost by indexing depth",
        fontsize=12)

    cost_x = ["baseline\n(no Acropolis,\nGlob/Grep/Read)",
              "Acropolis L0\n(name only)",
              "Acropolis L1\n(metadata)",
              "Acropolis L2\n(LLM summary)"]
    cost_vals = [base_total / 1000,
                 est_total[0] / 1000,
                 est_total[1] / 1000,
                 est_total[2] / 1000]
    colors = ["#888", "#fbb479", "#7ed996", "#9c8aff"]
    bars = ax.bar(cost_x, cost_vals, color=colors,
                  edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, cost_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 5,
                f"{v:.0f}K",
                ha="center", va="bottom", fontsize=10, weight="bold")
    ax.set_ylabel(f"estimated tokens, {N_QUERIES} queries (thousands)")
    ax.set_title(
        f"{N_QUERIES} queries, single-config session\n"
        "(model: constant overhead + 15 x payload + cache propagation)",
        fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.tick_params(axis="x", labelsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(HERE, "fresh_fig6_depth_cost_estimate.png"),
                dpi=140, bbox_inches="tight")
    print("wrote fresh_fig6_depth_cost_estimate.png")
    print(f"  est totals: L0={est_total[0]:,.0f}  L1={est_total[1]:,.0f}  "
          f"L2={est_total[2]:,.0f}  baseline={base_total:,.0f}")


def fig5_per_query_tokens(base, cell, arm1_pq, arm2_pc_total, arm2_pc_perq):
    """Per-query token cost: baseline vs Acropolis (best fixed config).

    Two panels:
      A: 15 paired bars (baseline vs Acropolis-L2-best) per query.
      B: cumulative total across 15 queries.
    """
    # Best fixed config (consistent with fig3)
    config_score = []
    for b in BACKENDS:
        for lv in LEVELS:
            hits = sum(cell[(b, lv)])
            tok_per_q = arm2_pc_total[(b, lv)] / N_QUERIES
            config_score.append((hits, -tok_per_q, b, lv))
    config_score.sort(reverse=True)
    best_b, best_lv = config_score[0][2], config_score[0][3]

    base_per_q = [arm1_pq.get(q, 0) for q in range(1, N_QUERIES + 1)]
    # Real per-query Acropolis cost = the (q, best_b, best_lv) entry in per_config.
    # NOTE: this is 1/21 of that query's turn usage in arm 2 (the agent batched
    # 21 configs into one turn per query). It approximates what a single-config
    # user would pay, but slightly overstates because arm 2's conversation
    # history was 21x bigger than a single-config session would have.
    acro_per_q = [arm2_pc_perq[(best_b, best_lv)].get(q, 0)
                  for q in range(1, N_QUERIES + 1)]

    base_total = sum(base_per_q)
    acro_total = sum(acro_per_q)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle(
        f"Per-query token cost: baseline vs Acropolis "
        f"({best_b}-L{best_lv}, single sem_q call)",
        fontsize=12)

    # Panel A: per-query bars
    x = np.arange(N_QUERIES)
    w = 0.40
    axes[0].bar(x - w / 2, [v / 1000 for v in base_per_q], w,
                label="baseline (Glob/Grep/Read)",
                color="#888", edgecolor="black", linewidth=0.4)
    axes[0].bar(x + w / 2, [v / 1000 for v in acro_per_q], w,
                label=f"Acropolis ({best_b}-L{best_lv})",
                color="#9c8aff", edgecolor="black", linewidth=0.4)
    # Annotate bars with values
    for i in range(N_QUERIES):
        axes[0].text(i - w / 2, base_per_q[i] / 1000 + 1,
                     f"{base_per_q[i] / 1000:.0f}",
                     ha="center", va="bottom", fontsize=7)
        axes[0].text(i + w / 2, acro_per_q[i] / 1000 + 1,
                     f"{acro_per_q[i] / 1000:.0f}",
                     ha="center", va="bottom", fontsize=7, color="purple")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"q{i + 1}" for i in range(N_QUERIES)],
                            fontsize=9)
    axes[0].set_ylabel("tokens per query (thousands, billable-equiv)")
    axes[0].set_xlabel("query")
    axes[0].legend(loc="upper left", fontsize=10)
    axes[0].grid(axis="y", linestyle=":", alpha=0.4)
    axes[0].set_title("Per-query token cost (real, mined from session JSONL)")

    # Panel B: cumulative total
    cfgs_b = ["baseline\n(Glob/Grep/Read)",
              f"Acropolis\n({best_b}-L{best_lv})"]
    tot_vals = [base_total / 1000, acro_total / 1000]
    bars = axes[1].bar(cfgs_b, tot_vals,
                       color=["#888", "#9c8aff"],
                       edgecolor="black", linewidth=0.4)
    for b, v in zip(bars, tot_vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v * 1.02,
                     f"{v:.0f}K", ha="center", va="bottom", fontsize=11,
                     weight="bold")
    saved = (base_total - acro_total) / base_total * 100
    axes[1].set_ylabel(f"cumulative tokens, {N_QUERIES} queries (thousands)")
    axes[1].set_title(f"Total ({N_QUERIES} queries)\n"
                      f"Acropolis saves {saved:.0f}% tokens")
    axes[1].grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(HERE, "fresh_fig5_per_query_tokens.png"),
                dpi=140, bbox_inches="tight")
    print("wrote fresh_fig5_per_query_tokens.png")


def fig4_pareto(base, cell, arm1_pq, arm2_pc_total):
    """Accuracy vs token-cost scatter. Each Acropolis config = one point.
    Baseline is also plotted. L0 -> L1 -> L2 connected per backend."""
    base_hits = sum(1 for v in base.values() if v["hit"])
    base_mean_tok = sum(arm1_pq.values()) / N_QUERIES

    fig, ax = plt.subplots(figsize=(10, 7))

    backend_color = plt.cm.tab10(np.linspace(0, 1, len(BACKENDS)))
    level_marker = {0: "o", 1: "s", 2: "D"}
    level_size = {0: 80, 1: 110, 2: 160}

    for bi, b in enumerate(BACKENDS):
        xs, ys = [], []
        for lv in LEVELS:
            hits = sum(cell[(b, lv)])
            tok = arm2_pc_total[(b, lv)] / N_QUERIES
            xs.append(hits)
            ys.append(tok / 1000)
            ax.scatter(hits, tok / 1000, color=backend_color[bi],
                       marker=level_marker[lv], s=level_size[lv],
                       edgecolor="black", linewidth=0.7,
                       label=f"{b} L{lv}" if False else None, zorder=3)
            ax.annotate(f"L{lv}", (hits, tok / 1000),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color=backend_color[bi])
        ax.plot(xs, ys, color=backend_color[bi], alpha=0.5, linewidth=1.2,
                zorder=2, label=b)

    # Baseline point
    ax.scatter(base_hits, base_mean_tok / 1000, color="black", marker="*",
               s=350, edgecolor="white", linewidth=1.2, zorder=4,
               label="baseline (Glob/Grep/Read)")
    ax.annotate("baseline", (base_hits, base_mean_tok / 1000),
                xytext=(8, 6), textcoords="offset points", fontsize=10,
                weight="bold")

    # "Win zone" shading: lower-right (high accuracy, low tokens)
    ax.axhline(base_mean_tok / 1000, color="gray", linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.axvline(base_hits, color="gray", linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.text(N_QUERIES * 0.98, base_mean_tok / 1000 * 0.5,
            "lower tokens than baseline\n(Acropolis is cheaper here)",
            ha="right", fontsize=8, color="darkgreen", style="italic")

    ax.set_xlabel(f"queries correct (of {N_QUERIES})")
    ax.set_ylabel("mean tokens/query (thousands, billable-equiv)")
    ax.set_xlim(0, N_QUERIES + 1)
    ax.set_ylim(0, max(base_mean_tok / 1000 * 1.2,
                       max(arm2_pc_total.values()) / N_QUERIES / 1000 * 1.1))
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title(
        f"Accuracy vs token cost — {N_QUERIES} queries\n"
        "(each colored line = one backend; markers = L0/L1/L2; "
        "star = baseline)",
        fontsize=11)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fresh_fig4_pareto.png"),
                dpi=140, bbox_inches="tight")
    print("wrote fresh_fig4_pareto.png")


if __name__ == "__main__":
    base, cell = grade()
    print("\n=== baseline ===")
    for qid, v in base.items():
        print(f"  q{qid:>2}: hit={v['hit']}  calls={v['n_calls']}")
    print(f"  total: {sum(1 for v in base.values() if v['hit'])}/{N_QUERIES}  "
          f"calls={sum(v['n_calls'] for v in base.values())}")

    print(f"\n=== Acropolis matrix (hits/{N_QUERIES} per (backend, level)) ===")
    print(f"{'backend':<20} {'L0':>4} {'L1':>4} {'L2':>4}")
    for b in BACKENDS:
        print(f"{b:<20} "
              f"{sum(cell[(b,0)]):>4} "
              f"{sum(cell[(b,1)]):>4} "
              f"{sum(cell[(b,2)]):>4}")

    arm1_pq, arm2_pq, arm2_pc_total, arm2_pc_perq = load_tokens()
    print(f"\n=== token cost (billable-equiv, mined from session JSONLs) ===")
    print(f"  arm1 mean tokens/query: "
          f"{sum(arm1_pq.values()) / N_QUERIES:>10,.0f}")
    print(f"  arm2 per-config mean tokens/query (across 15 queries):")
    print(f"  {'backend':<20} {'L0':>10} {'L1':>10} {'L2':>10}")
    for b in BACKENDS:
        row = [arm2_pc_total[(b, lv)] / N_QUERIES for lv in LEVELS]
        print(f"  {b:<20} {row[0]:>10,.0f} {row[1]:>10,.0f} {row[2]:>10,.0f}")

    fig1_heat(base, cell)
    fig2_per_query(base, cell)
    fig3_summary(base, cell, arm1_pq, arm2_pc_total)
    fig4_pareto(base, cell, arm1_pq, arm2_pc_total)
    fig5_per_query_tokens(base, cell, arm1_pq, arm2_pc_total, arm2_pc_perq)
    fig6_depth_cost_estimate(arm1_pq, arm2_pc_total)
