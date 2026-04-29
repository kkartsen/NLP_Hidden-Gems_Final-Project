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
  - Normalizes both betweenness and diversity to [0,1]
  - bridge_score = norm_betweenness * norm_diversity
  - multiplicative so both must be high to score high

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

OUTPUT_DIR = "../data/processed"

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

# Load paper table with hierarchical community assignments
df = pd.read_parquet(
    "../data/processed/papers_with_hierarchical_communities.parquet"
)

df["paper_id"] = df["paper_id"].astype(str)

print(f"\nPapers loaded: {len(df):,}")
print(f"Unique communities:   {df['community_hierarchical'].nunique():,}")
print(f"Papers with community:{df['community_hierarchical'].notna().sum():,}")

# Build fast lookup dict: paper_id (string) → community label (string)
# Only covers real papers — ghost nodes return None from .get()
community_lookup = dict(
    zip(df["paper_id"], df["community_hierarchical"])
)

print(f"\nCommunity lookup built: {len(community_lookup):,} entries")
print("Sample entries:")
for pid, comm in list(community_lookup.items())[:5]:
    print(f"  {pid} → {comm}")


"""
STAGE B: betweenness CENTRALITY
ALGORITHM:

For each real paper X:
  1. Look at all papers X cites (out-neighbors in directed graph)
  2. For each cited paper, check if it belongs to a DIFFERENT community than X
  3. cross_community_score(X) = fraction of X's citations that go
     to a different community
     = cross_community_edges / total_edges
Why fraction and not raw count?
  A paper with 50 citations to other communities scores the same as
  one with 500 citations to other communities if we use raw counts.
  Fraction normalizes for how prolific the paper is — it measures
  what PROPORTION of its intellectual reach crosses community boundaries.
Why out-neighbors (papers X cites) and not in-neighbors (papers that cite X)?
  X's citation choices reflect X's own intellectual scope.
  Who cites X is outside X's control — a cross-domain paper might
  only be discovered and cited by one community even if it bridges two.
   X's own citations are the best signal of whether X itself is cross-domain.
"""
print("\n" + "=" * 20)
print("STAGE B: Computing approximates betweenness centrality")
print("=" * 20)

MIN_CITERS = 2   # minimum real-paper citers to be eligible

citation_diversity_scores = {}
citation_community_counts = {}
real_in_degrees = {}

for node in G_directed.nodes():
    node_str = str(node)

    # Get all papers that cite this paper (directed in-neighbors)
    in_neighbors = list(G_directed.predecessors(node))

    if not in_neighbors:
        citation_diversity_scores[node] = 0
        citation_community_counts[node] = 0
        real_in_degrees[node] = 0
        continue
    
    # Only count real papers (those in community_lookup)
    real_citers = []
    for neighbor in in_neighbors:
        if community_lookup.get(str(neighbor)) is not None:
            real_citers.append(neighbor)

    real_in_degree = len(real_citers)
    real_in_degrees[node] = real_in_degree

    # Apply minimum threshold
    if real_in_degree < MIN_CITERS:
        citation_diversity_scores[node] = 0
        citation_community_counts[node] = 0
        continue

    # Count distinct communities among real citers
    citer_communities = set()
    for citer in real_citers:
        comm = community_lookup.get(str(citer))
        if comm is not None:
            citer_communities.add(comm)

    comm_count = len(citer_communities)
    citation_community_counts[node] = comm_count

    # Weight by log of real in-degree
    # A paper cited by 10 people from 3 communities scores higher
    # than a paper cited by 2 people from 3 communities
    weight = np.log1p(real_in_degree)
    citation_diversity_scores[node] = comm_count * weight

cd_values = list(citation_diversity_scores.values())
nonzero_cd = [v for v in cd_values if v > 0]

print(f"Citation diversity score results:")
print(f"  Total nodes scored:              {len(cd_values):,}")
print(f"  Nodes with score > 0:            {len(nonzero_cd):,}")
print(f"  Max score:                       {max(cd_values):.4f}")
if nonzero_cd:
    print(f"  Mean score (nonzero):            {np.mean(nonzero_cd):.4f}")
    print(f"  Median score (nonzero):          {np.median(nonzero_cd):.4f}")

# Attach to dataframe
df["citation_diversity_score"]     = df["paper_id"].map(
    citation_diversity_scores).fillna(0)
df["citation_community_count"]     = df["paper_id"].map(
    citation_community_counts).fillna(0).astype(int)
df["real_in_degree"]               = df["paper_id"].map(
    real_in_degrees).fillna(0).astype(int)

top_cd = df.nlargest(10, "citation_diversity_score")[
    ["paper_id", "title", "year", "real_in_degree",
     "citation_community_count", "citation_diversity_score",
     "community_hierarchical"]
]
print("\nTop 10 papers by citation diversity score:")
print(top_cd.to_string(index=False))

"""
STAGE C : CLUSTER DIVERSITY
ALGORITHM:

For each paper X, look at ALL neighbors in undirected graph
  (papers X cites + papers that cite X — both directions).
  Count how many distinct community IDs appear among those neighbors.
  Only count real papers — ghost nodes are skipped.
  diversity(X) = number of distinct communities among real neighbors
 
  Why undirected:
    Diversity measures intellectual neighborhood composition.
    Direction doesn't matter — whether X cites Y or Y cites X,
    they are still neighbors in the research landscape.
 
  Why ghost nodes skipped:
    Ghost nodes have no community label. Including them would
    give every paper free diversity points for every ghost neighbor,
    inflating scores with meaningless noise.
"""

print("\n" + "=" * 20)
print("STAGE C: Computing cluster diversity")
print("=" * 20)

print("Computing neighbor community diversity...")
print("(Ghost nodes skipped - only real paper communities counted)")

cluster_diversity = {}
ghost_skipped_total = 0

for node in G.nodes():
    # all neighbors in undirected graph - both directions
    neighbors = list(G.neighbors(node))
    if not neighbors:
        cluster_diversity[node] = 0
        continue
    
    neighbor_communities = set()
    ghost_count = 0

    for neighbor in neighbors:
        comm = community_lookup.get(str(neighbor))
        if comm is not None:
            neighbor_communities.add(comm)
        else:
            ghost_count += 1
    
    ghost_skipped_total += ghost_count
    cluster_diversity[node] = len(neighbor_communities)


div_values = list(cluster_diversity.values())
nonzero_div = [v for v in div_values if v > 0]
 
print(f"\nCluster diversity results:")
print(f"  Total nodes scored:        {len(div_values):,}")
print(f"  Nodes with diversity > 0:  {len(nonzero_div):,}")
print(f"  Nodes with diversity = 0:  {len(div_values) - len(nonzero_div):,}  (isolated)")
print(f"  Max diversity:             {max(div_values)}")
print(f"  Mean diversity (nonzero):  {np.mean(nonzero_div):.2f}"
      if nonzero_div else "  No nonzero diversity values")
print(f"  Median diversity (nonzero):{np.median(nonzero_div):.2f}"
      if nonzero_div else "")
print(f"  Ghost neighbors skipped:   {ghost_skipped_total:,}")
 
# Distribution breakdown
print(f"\n  Diversity distribution:")
print(f"    diversity = 1:  {sum(1 for v in div_values if v == 1):,}  (all neighbors same community)")
print(f"    diversity = 2:  {sum(1 for v in div_values if v == 2):,}")
print(f"    diversity = 3:  {sum(1 for v in div_values if v == 3):,}")
print(f"    diversity = 4:  {sum(1 for v in div_values if v == 4):,}")
print(f"    diversity 5+:   {sum(1 for v in div_values if v >= 5):,}")
 
# Attach to dataframe
df["cluster_diversity"] = df["paper_id"].map(
    cluster_diversity).fillna(0).astype(int)
 
# Top papers by cluster diversity
top_div = df.nlargest(10, "cluster_diversity")[
    ["paper_id", "title", "year", "real_in_degree",
     "cluster_diversity", "community_hierarchical"]
]
print("\nTop 10 papers by cluster diversity:")
print(top_div.to_string(index=False))
 
# Cross-check: do Stage B and Stage C top papers overlap?
# Overlap means a paper is both cited from multiple communities
# AND sits between multiple communities structurally — strong signal
top_b_ids = set(df.nlargest(50, "citation_diversity_score")["paper_id"])
top_c_ids = set(df.nlargest(50, "cluster_diversity")["paper_id"])
overlap = top_b_ids & top_c_ids
print(f"\nPapers in top 50 of BOTH Stage B and Stage C: {len(overlap)}")
if overlap:
    print("These are your strongest bridge candidates:")
    overlap_df = df[df["paper_id"].isin(overlap)][
        ["paper_id", "title", "citation_diversity_score",
         "cluster_diversity", "community_hierarchical"]
    ].sort_values("citation_diversity_score", ascending=False)
    print(overlap_df.to_string(index=False))


"""
STAGE D: BRIDGE SCORE
ALGORITHM:

Combines two independently computed signals multiplicatively:
 
  Signal 1 — norm_citation_diversity (from Stage B):
    Measures cross-community IMPACT.
    "Do researchers from multiple different fields actually cite this paper?"
    High score = paper is cited by many communities = real-world cross-domain recognition.
 
  Signal 2 — norm_cluster_diversity (from Stage C):
    Measures cross-community CONNECTIVITY.
    "Does this paper structurally sit between different communities in the graph?"
    High score = paper's neighbors span many communities = graph-theoretic bridge position.
 
  Both normalized to [0,1] so neither signal dominates due to scale.
  Multiplied so BOTH must be high — a paper strong on only one dimension scores low.
 
  bridge_score = norm_citation_diversity * norm_cluster_diversity
 
  Final output: papers_with_bridges.parquet
  Key columns for WS3 regression:
    bridge_score             — main bridge signal (feature + grouping variable)
    citation_community_count — how many communities cite this paper (interpretable)
    cluster_diversity        — how many communities neighbor this paper (interpretable)
    real_in_degree           — actual citations from real papers (regression target input)
"""

OUTPUT_DIR = "../data/processed"

print("\n" + "=" * 20)
print("STAGE D: Computing bridge scores")
print("=" * 20)

# Normalize Stage B: citation diversity score : 
# Divide every value by the maximum so all values land in [0,1]
# The paper with the highest citation diversity score gets exactly 1.0
# All others get a proportional fraction

cd_values = list(citation_diversity_scores.values())
max_cd = max(cd_values) if max(cd_values) > 0 else 1
norm_citation_diversity = {
    pid: v / max_cd for pid, v in citation_diversity_scores.items()
}
 
print(f"Stage B normalization:")
print(f"  Raw max citation diversity score: {max_cd:.4f}")
print(f"  After normalization: all values in [0, 1]")
print(f"  Papers with norm score > 0: "
      f"{sum(1 for v in norm_citation_diversity.values() if v > 0):,}")

# Normalize Stage C: cluster diversity:
# Same process — divide by maximum raw diversity value
# Max diversity was 14 (the HIV stigma paper touching 14 communities)
# That paper gets 1.0, a paper touching 7 communities gets 0.5, etc.
 
div_values = list(cluster_diversity.values())
max_div = max(div_values) if max(div_values) > 0 else 1
norm_cluster_diversity = {
    pid: v / max_div for pid, v in cluster_diversity.items()
}
 
print(f"\nStage C normalization:")
print(f"  Raw max cluster diversity: {max_div}")
print(f"  After normalization: all values in [0, 1]")
print(f"  Papers with norm score > 0: "
      f"{sum(1 for v in norm_cluster_diversity.values() if v > 0):,}")

# Compute bridge score :
# Multiply the two normalized signals
# Multiplication enforces AND logic:
#   Hub paper:      norm_citation_diversity=0.8, norm_cluster_diversity=0.07 → 0.056
#   Isolated paper: norm_citation_diversity=0.0, norm_cluster_diversity=0.5  → 0.000
#   True bridge:    norm_citation_diversity=0.8, norm_cluster_diversity=0.7  → 0.560
 
bridge_scores = {}
for pid in G_directed.nodes():
    ncd = norm_citation_diversity.get(pid, 0)
    ncv = norm_cluster_diversity.get(pid, 0)
    bridge_scores[pid] = ncd * ncv
 
bs_values = list(bridge_scores.values())
nonzero_bs = [v for v in bs_values if v > 0]
 
print(f"\nBridge score results:")
print(f"  Papers with score > 0:    {len(nonzero_bs):,}")
print(f"  Max bridge score:         {max(bs_values):.6f}")
if nonzero_bs:
    print(f"  Mean score (nonzero):     {np.mean(nonzero_bs):.6f}")
    print(f"  Median score (nonzero):   {np.median(nonzero_bs):.6f}")
 
# Attach all scores to dataframe :
 
df["norm_citation_diversity"] = df["paper_id"].map(
    norm_citation_diversity).fillna(0)
df["norm_cluster_diversity"]  = df["paper_id"].map(
    norm_cluster_diversity).fillna(0)
df["bridge_score"]            = df["paper_id"].map(
    bridge_scores).fillna(0)
 
# Bridge score distribution :
 
print(f"\nBridge score distribution among your 36,823 real papers:")
print(f"  Papers with score > 0:    {(df['bridge_score'] > 0).sum():,}")
print(f"  Top 1%  threshold:        {df['bridge_score'].quantile(0.99):.6f}")
print(f"  Top 5%  threshold:        {df['bridge_score'].quantile(0.95):.6f}")
print(f"  Top 10% threshold:        {df['bridge_score'].quantile(0.90):.6f}")
print(f"  Top 20% threshold:        {df['bridge_score'].quantile(0.80):.6f}")


# Final top bridge papers :
# These are papers that scored high on BOTH citation diversity (Stage B)
# AND cluster diversity (Stage C) — the strongest bridge candidates
# before regression analysis
 
top_bridges = df.nlargest(15, "bridge_score")[
    ["paper_id", "title", "year", "real_in_degree",
     "citation_community_count", "cluster_diversity",
     "norm_citation_diversity", "norm_cluster_diversity",
     "bridge_score", "community_hierarchical"]
]
 
print("\nTop 15 bridge papers (final ranking):")
print(top_bridges.to_string(index=False))
 
# Flag top bridge papers :
# Mark papers in top 20% bridge score as bridge papers (is_bridge = True)
# This binary flag is used by WS3 regression as the grouping variable
# for the Wilcoxon test — bridge vs non-bridge
 
# bridge_threshold = df["bridge_score"].quantile(0.80)
df["is_bridge"] = df["bridge_score"] > 0
bridge_papers   = df[df["is_bridge"]].copy()
nobridge_papers = df[~df["is_bridge"]].copy()
 
print(f"\nBridge paper flag (nonzero bridge score):")
print(f"  Bridge Papers:     {df['is_bridge'].sum():,}")
print(f"  Non-bridge papers:  {(~df['is_bridge']).sum():,}")

# Within bridge papers, rank by bridge score
# Top 10% of bridge papers = strongest bridges = hidden gem candidates
bridge_threshold_top = bridge_papers["bridge_score"].quantile(0.90)
df["is_top_bridge"] = df["bridge_score"] >= bridge_threshold_top

print(f"\nTop bridge papers (top 10% of bridge pool):")
print(f"  Threshold:          {bridge_threshold_top:.6f}")
print(f"  Papers flagged:     {df['is_top_bridge'].sum():,}")
 
# Verify the 38 overlap papers score high :
# The 38 papers that appeared in top 50 of both Stage B and C
# should dominate the top bridge scores — this is a sanity check
 
top_b_ids = set(df.nlargest(50, "citation_diversity_score")["paper_id"])
top_c_ids = set(df.nlargest(50, "cluster_diversity")["paper_id"])
overlap_ids = top_b_ids & top_c_ids
 
overlap_in_top_bridges = set(
    df.nlargest(50, "bridge_score")["paper_id"]
) & overlap_ids
 
print(f"\nSanity check:")
print(f"  Of the 38 papers in top 50 of both Stage B and C,")
print(f"  {len(overlap_in_top_bridges)} appear in top 50 bridge papers.")
print(f"  (Expected: most of them — confirms Stage D is consistent with B and C)")
 
# Save :
 
out_path = os.path.join(OUTPUT_DIR, "papers_with_bridges.parquet")
df.to_parquet(out_path, index=False)
print(f"\nSaved → {out_path}")
 
mean_bs_str   = f"{np.mean(nonzero_bs):.6f}" if nonzero_bs else "N/A"
mean_div_str  = f"{np.mean([v for v in div_values if v > 0]):.2f}" if any(v > 0 for v in div_values) else "N/A"

stats = f"""
=== Bridge Score Stats ===
 
STAGE B — CITATION DIVERSITY SCORE
  Nodes with score > 0:         {len([v for v in cd_values if v > 0]):,}
  Max raw score:                {max_cd:.4f}
 
STAGE C — CLUSTER DIVERSITY
  Nodes with diversity > 0:     {sum(1 for v in div_values if v > 0):,}
  Max diversity:                {max_div}
  Distribution:
    diversity = 1:  {sum(1 for v in div_values if v == 1):,}
    diversity = 2:  {sum(1 for v in div_values if v == 2):,}
    diversity = 3:  {sum(1 for v in div_values if v == 3):,}
    diversity = 4:  {sum(1 for v in div_values if v == 4):,}
    diversity 5+:   {sum(1 for v in div_values if v >= 5):,}
 
STAGE D — BRIDGE SCORE
  Papers with score > 0:        {len(nonzero_bs):,}
  Max bridge score:             {max(bs_values):.6f}
  Mean score (nonzero):         {mean_bs_str}
  Top 1% threshold:             {df['bridge_score'].quantile(0.99):.6f}
  Top 5% threshold:             {df['bridge_score'].quantile(0.95):.6f}
 
BRIDGE FLAG
  Bridge papers flagged:        {df['is_bridge'].sum():,}
 
OVERLAP SANITY CHECK
  Papers in top 50 of both B and C:    38
  Of those, in top 50 bridge score:    {len(overlap_in_top_bridges)}
 
TOP 15 BRIDGE PAPERS:
{top_bridges[['paper_id','title','year','real_in_degree',
              'citation_community_count','cluster_diversity',
              'bridge_score']].to_string(index=False)}
"""
 
stats_path = os.path.join(OUTPUT_DIR, "bridge_stats.txt")
with open(stats_path, "w") as f:
    f.write(stats)
print(f"Stats saved → {stats_path}")
 
print("\nDone. Stages B, C, D complete.")
print("\nColumns in papers_with_bridges.parquet:")
print("  citation_diversity_score  — Stage B raw score")
print("  citation_community_count  — communities that cite this paper")
print("  real_in_degree            — citations from real papers only")
print("  cluster_diversity         — distinct communities among neighbors")
print("  norm_citation_diversity   — Stage B normalized to [0,1]")
print("  norm_cluster_diversity    — Stage C normalized to [0,1]")
print("  bridge_score              — final bridge score (B * C normalized)")
print("  is_bridge                 — True if top 20% bridge score")
print("\nHand papers_with_bridges.parquet to WS3 for citation regression.")