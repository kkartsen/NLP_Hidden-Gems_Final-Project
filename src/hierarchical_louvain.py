"""
Splits oversized communities into finer sub-communities using a second pass of Louvain

Problem:
Standard louvain at any resolution produces one giant community containing ~45% of all papers.
This is too coarse for bridge detection, if half the dataset is one "community", crossing into it means nothing

Solution:
Level 1: Global Louvain on full graph(already computed)
- finds broad research areas 

Level 2: for each community above SIZE_THRESHOLD papers, extract its subgraph and run louvain again 
at higher resolution
- finds specific sub-topics within each broad area

final community ID = hierarchical label, e.g., community 3 -> sub-communities 3_0, 3_1

small communities stay as-is: community 7 stays 7

Why this works:
  The giant community IS internally structured — it contains HIV papers,
  immunology papers, epidemiology papers — but their cross-citations
  make global Louvain lump them together. By zooming into just that
  subgraph, Louvain can find the finer internal structure without
  being pulled by connections to the rest of the graph.

Why higher resolution for Level 2?
  Inside a dense subgraph, we WANT smaller communities.
  Resolution 2.0-3.0 pushes Louvain to find tighter, more specific clusters.
 
Inputs:
  ../data/processed/citation_graph.pkl
  ../data/processed/papers_with_bridges.parquet  (has community_id from Level 1)
 
Outputs:
  ../data/processed/papers_with_hierarchical_communities.parquet
  ../data/processed/hierarchical_community_stats.txt
"""

import pickle
import pandas as pd
import networkx as nx
import community as community_louvain
from collections import Counter
import numpy as np
import os

OUTPUT_DIR = "../data/processed"

# communities larger than this gets in level 2
SIZE_THRESHOLD = 800
LEVEL2_RESOLUTION = 2.5

# load graph and existing community assignments
print("Loading graph...")
with open("../data/processed/citation_graph.pkl","rb") as f:
    G_directed = pickle.load(f)

G = G_directed.to_undirected()
print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

print("\nLoading Level 1 community assignments...")
df = pd.read_parquet("../data/processed/papers.parquet")
df["paper_id"] = df["paper_id"].astype(str)

#rebuild the full partition dict (paperID -> communityID)
#re-run level 1 louvain to get the full partition on including ghost nodes
#paper.parquet only has 36,823 real papers
print("Re-running Level 1 Louvain to get full partition (including ghost nodes)...")
partition_L1 = community_louvain.best_partition(
    G,
    resolution=1.0,
    random_state=42
)
print(f"Level 1 {len(set(partition_L1.values()))} communities")

# Identify oversized communities
community_sizes = Counter(partition_L1.values())
oversized = [cid for cid, size in community_sizes.items() if size>SIZE_THRESHOLD]

print(f"\nCommunities above threshold ({SIZE_THRESHOLD} papers): {len(oversized)}")
for cid in oversized:
    print(f" Community {cid}: {community_sizes[cid]:,} papers")

# LEVEL 2 Louvain on each oversized community C
# 1. Find all nodes in C: nodes in C = {node for node, comm in partition if comm == C}
# 2. Extract subgraph: G_sub = subgraph induced by nodes_in_C
#    "Induced" means: keep only nodes in C, keep only edges where BOTH
#    endpoints are in C. Edges going outside C are cut — we want
#    internal structure, not external connections pulling things together.
# 3. Run Louvain on G_sub at higher resolution → sub-partition
# 4. Assign new hierarchical IDs: "{parent_community}_{sub_community}"
#    e.g. community 0 splits into "0_0", "0_1", "0_2", "0_3" ...

print(f"\nRunning Level 2 Louvain (resolution={LEVEL2_RESOLUTION})...")

# we'll overwrite oversized communities
final_partition = dict(partition_L1)  # node → community label (will become strings)

# Convert all Level 1 labels to strings first for consistency
final_partition = {node: str(comm) for node, comm in final_partition.items()}

for parent_comm in oversized:
    # Step 1: find all nodes belonging to this community
    nodes_in_comm = [node for node, comm in partition_L1.items()
                     if comm == parent_comm]
 
    print(f"\n  Community {parent_comm} ({len(nodes_in_comm):,} nodes) — splitting...")
 
    # Step 2: extract the induced subgraph
    # G.subgraph(nodes) returns a VIEW — a read-only subgraph
    # containing only those nodes and edges between them
    G_sub = G.subgraph(nodes_in_comm)
 
    print(f"    Subgraph: {G_sub.number_of_nodes():,} nodes, "
          f"{G_sub.number_of_edges():,} edges")
    
    # Step 3: run Louvain on the subgraph
    # If the subgraph has no edges (all nodes disconnected within community),
    # skip — we can't split further
    if G_sub.number_of_edges() == 0:
        print(f"    No internal edges — cannot split, keeping as community {parent_comm}")
        continue
 
    sub_partition = community_louvain.best_partition(
        G_sub,
        resolution=LEVEL2_RESOLUTION,
        random_state=42
    )
 
    sub_sizes = Counter(sub_partition.values())
    print(f"    Sub-communities found: {len(sub_sizes)}")
    print(f"    Largest sub-community: {max(sub_sizes.values())} papers")
    print(f"    Smallest sub-community: {min(sub_sizes.values())} papers")

    # Step 4: assign hierarchical labels back to the main partition
    # Format: "{parent_community_id}_{sub_community_id}"
    # e.g. if parent is community 0, sub-communities become "0_0", "0_1", ...
    for node, sub_comm in sub_partition.items():
        final_partition[node] = f"{parent_comm}_{sub_comm}"

# Assess final partition
print("\n" + "=" * 20)
print("FINAL HIERARCHICAL PARTITION SUMMARY")
print("=" * 20)

final_sizes = Counter(final_partition.values())
all_sizes = sorted(final_sizes.values(), reverse=True)
n_final_communities = len(final_sizes)
 
print(f"Total communities (after splitting): {n_final_communities}")
print(f"Largest community:  {all_sizes[0]} papers")
print(f"Smallest community: {all_sizes[-1]} papers")
print(f"Median size:        {np.median(all_sizes):.0f} papers")
print(f"\nSize distribution:")
print(f"  2-10 papers:    {sum(1 for s in all_sizes if 2 <= s <= 10)}")
print(f"  11-100 papers:  {sum(1 for s in all_sizes if 11 <= s <= 100)}")
print(f"  101-1000 papers:{sum(1 for s in all_sizes if 101 <= s <= 1000)}")
print(f"  1000+ papers:   {sum(1 for s in all_sizes if s > 1000)}")


# ── Attach to dataframe (real papers only)
print("\nAttaching hierarchical community IDs to paper table...")
df["community_hierarchical"] = df["paper_id"].map(final_partition)
 
# How many of your real papers got assigned
assigned = df["community_hierarchical"].notna().sum()
print(f"Real papers with community assignment: {assigned:,} / {len(df):,}")
 
# Show sample of what communities look like now
print("\nSample community assignments:")
print(df[["paper_id", "title", "community_hierarchical"]].head(10).to_string(index=False))
 
# Top 10 largest communities among real papers only
real_comm_sizes = Counter(df["community_hierarchical"].dropna().tolist())
top10 = real_comm_sizes.most_common(10)
print("\nTop 10 largest communities (real papers only):")
for comm, size in top10:
    print(f"  Community {comm}: {size} papers")


out_path = os.path.join(OUTPUT_DIR, "papers_with_hierarchical_communities.parquet")
df.to_parquet(out_path, index=False)
print(f"\nSaved → {out_path}")
 
stats = f"""
=== Hierarchical Community Detection Stats ===
 
Level 1 Louvain (resolution=1.0):
  Communities found: {len(set(partition_L1.values()))}
 
Oversized communities split (threshold={SIZE_THRESHOLD}):
  Count: {len(oversized)}
  IDs: {oversized}
 
Level 2 Louvain (resolution={LEVEL2_RESOLUTION}):
  Final community count: {n_final_communities}
  Largest community:     {all_sizes[0]} papers
  Median size:           {np.median(all_sizes):.0f} papers
 
Size distribution:
  2-10 papers:     {sum(1 for s in all_sizes if 2 <= s <= 10)}
  11-100 papers:   {sum(1 for s in all_sizes if 11 <= s <= 100)}
  101-1000 papers: {sum(1 for s in all_sizes if 101 <= s <= 1000)}
  1000+ papers:    {sum(1 for s in all_sizes if s > 1000)}
"""
 
stats_path = os.path.join(OUTPUT_DIR, "hierarchical_stats.txt")
with open(stats_path, "w") as f:
    f.write(stats)
print(f"Stats saved → {stats_path}")
 
print("\nDone. Use papers_with_hierarchical_communities.parquet for bridge scoring.")
print("Key new column: 'community_hierarchical'")
print("  Small communities keep integer label e.g. '7'")
print("  Split communities get hierarchical label e.g. '0_3'")