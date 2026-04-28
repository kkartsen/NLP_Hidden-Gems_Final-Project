import pickle
import time
import os
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────

GRAPH_PATH   = "data\processed\citation_graph.pkl"
PAPERS_PATH  = "data\processed\papers.parquet"
OUTPUT_DIR   = "data\processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOUVAIN_RESOLUTION = 1.5   # higher = more, smaller communities
LOUVAIN_SEED       = 42    # reproducibility
BC_K_SAMPLES       = 500   # number of pivot nodes for approximate BC
                           # k=500 is a good balance: fast + accurate for ranking
                           # reduce to k=200 if runtime is too long
TOP_N_GEMS         = 500   # how many top bridge papers to flag in the output


# ─────────────────────────────────────────────────────────────────────────────
# STAGE A: Load graph + run Louvain
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("STAGE A: Loading graph and running Louvain")
print("=" * 60)

t0 = time.time()

with open(GRAPH_PATH, "rb") as f:
    G_directed = pickle.load(f)

print(f"Directed graph:   {G_directed.number_of_nodes():,} nodes, "
      f"{G_directed.number_of_edges():,} edges")

# Convert to undirected — Louvain's modularity formula requires this.
# For citations, we care about *connection* not direction:
# "A cites B" and "B cites A" both mean the papers are related.
G = G_directed.to_undirected()
print(f"Undirected graph: {G.number_of_edges():,} edges "
      f"(some directed edges merge into one undirected edge)")


print(f"\nRunning Louvain (resolution={LOUVAIN_RESOLUTION}, seed={LOUVAIN_SEED})...")
partition = community_louvain.best_partition(
    G,
    resolution=LOUVAIN_RESOLUTION,
    random_state=LOUVAIN_SEED
)
# partition: {paper_id (str): community_id (int)}

n_communities = len(set(partition.values()))
modularity    = community_louvain.modularity(partition, G)

print(f"  Communities found : {n_communities:,}")
print(f"  Modularity Q      : {modularity:.4f}")

community_sizes = Counter(partition.values())
sizes = sorted(community_sizes.values(), reverse=True)
print(f"  Largest community : {sizes[0]:,} papers")
print(f"  Smallest community: {sizes[-1]:,} papers")
print(f"  Median size       : {np.median(sizes):.0f} papers")
print(f"  Singleton communities: {sum(1 for s in sizes if s == 1):,}")
print(f"  Stage A done in {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE B: Betweenness Centrality (approximate)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STAGE B: Betweenness centrality (approximate)")
print("=" * 60)



print(f"\nRunning approximate BC on undirected graph (k={BC_K_SAMPLES})...")
print("(This is fast on a sparse graph — expect < 60s)")

t1 = time.time()
betweenness = nx.betweenness_centrality(
    G,
    k=BC_K_SAMPLES,
    normalized=True,   # divides by (n-1)(n-2)/2 so score is in [0, 1]
    seed=LOUVAIN_SEED  # reproducibility
)
# betweenness: {paper_id: float}

elapsed_bc = time.time() - t1
print(f"  BC computed in {elapsed_bc:.1f}s")

bc_values = np.array(list(betweenness.values()))
print(f"  BC = 0 (isolated nodes) : {(bc_values == 0).sum():,}")
print(f"  BC > 0                  : {(bc_values > 0).sum():,}")
print(f"  BC max                  : {bc_values.max():.6f}")
print(f"  BC mean (non-zero)      : {bc_values[bc_values > 0].mean():.6f}")
print(f"  BC 95th percentile      : {np.percentile(bc_values, 95):.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE C: Cluster Diversity
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STAGE C: Cluster diversity")
print("=" * 60)



print("\nComputing cluster diversity for all nodes...")

t2 = time.time()
cluster_diversity = {}

for node in G.nodes():
    # G.neighbors(node) gives all neighbours in the undirected graph
    # (equivalent to: papers this node cites + papers that cite this node)
    neighbour_communities = {
        partition[nbr]
        for nbr in G.neighbors(node)
        if nbr in partition          # guard: all nodes should be in partition
    }
    # If a node has no neighbours (isolated), diversity = 1 (baseline)
    cluster_diversity[node] = max(len(neighbour_communities), 1)

elapsed_cd = time.time() - t2
print(f"  Diversity computed in {elapsed_cd:.1f}s")

div_values = np.array(list(cluster_diversity.values()))
print(f"  Diversity = 1 (single-community neighbours): {(div_values == 1).sum():,}")
print(f"  Diversity > 1 (cross-community neighbours) : {(div_values > 1).sum():,}")
print(f"  Max diversity : {div_values.max()}")
print(f"  Mean diversity: {div_values.mean():.2f}")
print(f"  95th percentile: {np.percentile(div_values, 95):.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE D: Bridge Score
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STAGE D: Bridge score = betweenness × cluster_diversity")
print("=" * 60)



print("\nComputing bridge scores...")

all_nodes = list(G.nodes())
bridge_scores     = {n: betweenness[n] * cluster_diversity[n] for n in all_nodes}
bridge_scores_log = {n: np.log1p(bridge_scores[n]) for n in all_nodes}

bs_values = np.array(list(bridge_scores.values()))
print(f"  Bridge score > 0  : {(bs_values > 0).sum():,}")
print(f"  Bridge score = 0  : {(bs_values == 0).sum():,}")
print(f"  Max bridge score  : {bs_values.max():.6f}")
print(f"  95th percentile   : {np.percentile(bs_values, 95):.8f}")

# Top 10 bridge papers (preview)
top10 = sorted(bridge_scores.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"\n  Top 10 bridge papers (paper_id → bridge_score):")
for pid, score in top10:
    title = G_directed.nodes[pid].get("title", "unknown")[:70]
    community = partition.get(pid, -1)
    diversity = cluster_diversity[pid]
    bc = betweenness[pid]
    print(f"    {pid}  score={score:.6f}  bc={bc:.6f}  div={diversity}")
    print(f"      \"{title}\"")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE E: Build output DataFrame
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STAGE E: Building output DataFrame")
print("=" * 60)

# Load the original papers table (metadata from WS1)
print("Loading papers.parquet...")
df = pd.read_parquet(PAPERS_PATH)
print(f"  Papers table: {df.shape[0]:,} rows × {df.shape[1]} cols")

# Build a per-node results frame from our computed scores
results = pd.DataFrame({
    "paper_id"         : list(all_nodes),
    "community_id"     : [partition[n] for n in all_nodes],
    "betweenness"      : [betweenness[n] for n in all_nodes],
    "cluster_diversity": [cluster_diversity[n] for n in all_nodes],
    "bridge_score"     : [bridge_scores[n] for n in all_nodes],
    "bridge_score_log" : [bridge_scores_log[n] for n in all_nodes],
})

# Merge with paper metadata on paper_id
# Left join on results so every node in the graph gets a row
df["paper_id"] = df["paper_id"].astype(str)
results["paper_id"] = results["paper_id"].astype(str)

merged = results.merge(df, on="paper_id", how="left")

# Flag the top bridge papers
bridge_threshold = merged["bridge_score"].quantile(1 - TOP_N_GEMS / len(merged))
merged["is_bridge_candidate"] = merged["bridge_score"] >= bridge_threshold

print(f"\n  Total rows in output: {len(merged):,}")
print(f"  Bridge candidates flagged: {merged['is_bridge_candidate'].sum():,}")
print(f"  Bridge score threshold used: {bridge_threshold:.8f}")
print(f"\n  Column list: {list(merged.columns)}")

# Sort by bridge score descending
merged = merged.sort_values("bridge_score", ascending=False).reset_index(drop=True)

# Preview top rows
print(f"\n  Top 5 bridge candidates:")
top_cols = ["paper_id", "community_id", "betweenness", "cluster_diversity",
            "bridge_score", "in_degree", "year", "title"]
available_cols = [c for c in top_cols if c in merged.columns]
print(merged[available_cols].head(5).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# STAGE F: Save outputs
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STAGE F: Saving outputs")
print("=" * 60)

# Main output for WS3 (regression team)
parquet_path = os.path.join(OUTPUT_DIR, "papers_with_bridges.parquet")
merged.to_parquet(parquet_path, index=False)
print(f"  papers_with_bridges.parquet → {parquet_path}")

# Also save a CSV of top bridge candidates for quick inspection / presentation
csv_path = os.path.join(OUTPUT_DIR, "top_bridge_papers.csv")
top_bridge_cols = ["paper_id", "title", "year", "journal", "community_id",
                   "betweenness", "cluster_diversity", "bridge_score",
                   "bridge_score_log", "in_degree"]
available_top_cols = [c for c in top_bridge_cols if c in merged.columns]
merged[merged["is_bridge_candidate"]][available_top_cols].to_csv(csv_path, index=False)
print(f"  top_bridge_papers.csv       → {csv_path}")

# Stats report
stats_report = f"""
=== Bridge Scoring Report ===

Graph
  Nodes        : {G.number_of_nodes():,}
  Edges        : {G.number_of_edges():,}

Stage A — Louvain
  Resolution   : {LOUVAIN_RESOLUTION}
  Communities  : {n_communities:,}
  Modularity Q : {modularity:.4f}
  Largest comm : {sizes[0]:,} papers
  Median comm  : {np.median(sizes):.0f} papers
  Singletons   : {sum(1 for s in sizes if s == 1):,}

Stage B — Betweenness Centrality
  k samples    : {BC_K_SAMPLES}
  Runtime      : {elapsed_bc:.1f}s
  Nodes BC > 0 : {(bc_values > 0).sum():,}
  BC max       : {bc_values.max():.6f}
  BC 95th pct  : {np.percentile(bc_values, 95):.6f}

Stage C — Cluster Diversity
  Runtime           : {elapsed_cd:.1f}s
  Max diversity     : {div_values.max()}
  Mean diversity    : {div_values.mean():.2f}
  Multi-community   : {(div_values > 1).sum():,} nodes

Stage D — Bridge Score
  Formula           : betweenness × cluster_diversity
  Nodes score > 0   : {(bs_values > 0).sum():,}
  Max bridge score  : {bs_values.max():.6f}
  95th percentile   : {np.percentile(bs_values, 95):.8f}
  Top candidates    : {merged['is_bridge_candidate'].sum():,}

Output files
  papers_with_bridges.parquet   — full table, use for regression (WS3)
  top_bridge_papers.csv         — top {TOP_N_GEMS} candidates, use for SciBERT (WS4)

Columns added to papers table:
  community_id      — Louvain cluster ID (integer)
  betweenness       — BC score, normalised [0, 1]
  cluster_diversity — number of distinct communities among neighbours
  bridge_score      — betweenness × cluster_diversity
  bridge_score_log  — log1p(bridge_score), better feature for regression
  is_bridge_candidate — True for top {TOP_N_GEMS} by bridge_score

WS3 (regression) uses:
  bridge_score_log, cluster_diversity, betweenness, in_degree, out_degree, year

WS4 (SciBERT) uses:
  top_bridge_papers.csv — run embeddings on abstract column
"""

stats_path = os.path.join(OUTPUT_DIR, "bridge_stats.txt")
with open(stats_path, "w") as f:
    f.write(stats_report)
print(f"  bridge_stats.txt            → {stats_path}")

print(stats_report)
print(f"\nDone. Total runtime: {time.time()-t0:.1f}s")
print("WS3 and WS4 can now load papers_with_bridges.parquet")
