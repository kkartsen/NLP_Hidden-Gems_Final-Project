"""
ALGORITHM DESIGN OVERVIEW
--------------------------
Stage A: Louvain community detection
  - Converts directed graph to undirected (Louvain requires undirected)
  - Runs Louvain at resolution=1.0 (can tune later)
  - Assigns every node a community integer ID
  - Modularity score tells us how good the clustering is (0 = random, 1 = perfect)

Stage B: Betweenness centrality (approximate)
  - For every node X, counts what fraction of shortest paths between
    random pairs (A,B) pass through X
  - Uses k=500 random samples (exact is O(VE) — too slow for 36k nodes)
  - Output: betweenness[paper_id] = float between 0 and 1

Stage C: Cluster diversity
  - For each paper, look at all its direct neighbors (papers it cites
    AND papers that cite it — both directions)
  - Count how many distinct community IDs appear among those neighbors
  - Output: diversity[paper_id] = integer (1 = all neighbors same community)

Stage D: Bridge score
  - bridge_score = betweenness * cluster_diversity
  - This penalizes pure hubs (high betweenness but low diversity)
    and rewards true bridges (high betweenness AND cross-community neighbors)

Inputs:
  data/processed/citation_graph.pkl
  data/processed/papers.parquet

Outputs:
  data/processed/papers_with_bridges.parquet  (main output for WS3)
  data/processed/bridge_stats.txt
"""

import pickle
import pandas as pd
import networkx as nx
import community as community_louvain
import numpy as np
from collections import Counter
import os

OUTPUT_DIR = "data/processed"


"""
STAGE 1: Louvain community detection
ALGORITHM:

Louvain works in two phases that repeat until modularity stops improving:
Phase 1 (local): each node tried moving to its neighbor's community. 
                 it keeps the move only if modularity increases.
                 sweeps all nodes repeatedly until no move helps.
Phase 2 (aggregate): collapse each community into a single "super-node"
                 edge weights between super-nodes = sum of edges between
                 their member nodes. then repeat phase 1 on this smaller graph

                 
Modularity Q = fraction of edges inside communities - expected fraction if edges were placed randomly
Q close to 1 = strong community structure
Q close to 0 = no real community structure (random graph)

Resolution parameter:
  Higher resolution → more communities, smaller size
  Lower resolution  → fewer communities, larger size
  Default 1.0 is the standard starting point

Why undirected?
 Louvain's modularity formula is defined for undirected graphs.
 For citations, we care about *connection*, not direction,
 when detecting communities. "A cites B" and "B cites A" both
 mean the papers are related — we treat them the same.
"""
print("=" * 20)
print("STAGE A: Loading Graph and running Louvain")
print("=" * 20)

with open("../data/processed/citation_graph.pkl", "rb") as f:
    G_directed = pickle.load(f)

print(f"Directed Graph: {G_directed.number_of_nodes():,} nodes, "
      f"{G_directed.number_of_edges():,} edges")

# convert to undirected for the louvain
G = G_directed.to_undirected()
print(f"Undirected Graph: {G.number_of_edges():,} edges")

# Run louvain
# random state = 42 makes results reproducible - same seed = same partition
RESOLUTION = 1.5
print(f"\nRunning Louvain (resolution={RESOLUTION}, random_state=42)...")

partition = community_louvain.best_partition(
    G,
    resolution=RESOLUTION,
    random_state=42
)

# partition is a dict: {paper_id: community_int}
# e.g. {'17396995': 3,...}

n_communities = len(set(partition.values()))
modularity = community_louvain.modularity(partition, G)

print(f"Communities found: {n_communities}")
print(f"Modularity score: {modularity:.4f}")
# Modularity interpretation:
#   > 0.3 is considered meaningful community structure
#   > 0.5 is strong
#   Our graph is sparse so expect a lower score

# Community size distribution — important diagnostic
community_sizes = Counter(partition.values())
sizes = sorted(community_sizes.values(), reverse=True)
print(f"Largest community: {sizes[0]} papers")
print(f"Smallest community: {sizes[-1]} papers")
print(f"Median community size: {np.median(sizes):.0f} papers")
print(f"Communities with only 1 paper (singletons): "
      f"{sum(1 for s in sizes if s == 1)}")