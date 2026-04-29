"""
Stage 5 - Statistical Test + Final Output + Visualization
WHAT THIS STAGE DOES
---------------------
1. Formally proves that bridge papers are systematically under-cited
   using the Wilcoxon signed-rank test
2. Produces the final ranked hidden gems table
3. Visualizes the citation network with hidden gems highlighted
 
ALGORITHM DESIGN
----------------
WHY WILCOXON AND NOT A T-TEST?
  The t-test assumes both groups follow a normal distribution.
  Citation residuals are heavily skewed — a few papers have very
  large positive residuals (massively over-cited) while most cluster
  near zero. This violates the t-test assumption.
 
  The Wilcoxon rank-sum test (Mann-Whitney U) makes NO distribution
  assumption. It works by:
    1. Pool all residuals from both groups (bridge + non-bridge)
    2. Rank them from smallest to largest (most under-cited = rank 1)
    3. Sum the ranks for the bridge group
    4. Ask: is this rank sum significantly lower than expected by chance?
  
  If bridge papers are systematically under-cited, their residuals
  will cluster at the low end of the ranked list — rank sum will be
  significantly lower than if bridge papers were randomly distributed.
 
  p < 0.05 means: there is less than 5% probability that this
  difference in rank sums happened by random chance.
  We can then claim: bridge papers are statistically significantly
  more under-cited than non-bridge papers.
 
EFFECT SIZE — CLIFF'S DELTA
  p-value tells you IF the effect is real.
  Effect size tells you HOW BIG the effect is.
  
  Cliff's delta = probability that a randomly chosen bridge paper
  has a lower residual than a randomly chosen non-bridge paper,
  minus the reverse probability.
  Range: -1 to 1
    |d| < 0.147 = negligible
    |d| < 0.330 = small
    |d| < 0.474 = medium
    |d| >= 0.474 = large
 
FINAL RANKING LOGIC
  Hidden gems are ranked by a combined score:
    final_score = bridge_semantic_score * abs(residual)
  
  This rewards papers that are:
    - Strongly confirmed as cross-domain by SPECTER (high semantic score)
    - Most under-cited relative to prediction (large negative residual)
  
  A paper with perfect semantic bridging but tiny under-citation ranks
  lower than a paper with strong semantic bridging AND severe under-citation.
 
Inputs:
  data/processed/papers_with_residuals.parquet
  data/processed/hidden_gems_candidates.parquet
  data/processed/citation_graph.pkl
 
Outputs:
  data/processed/hidden_gems_final.csv        (final deliverable)
  data/processed/wilcoxon_stats.txt           (statistical proof)
  data/processed/residual_distribution.png    (visualization 1)
  data/processed/network_visualization.html   (visualization 2)
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import networkx as nx
import os
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "../data/processed"

# LOAD DATA
print("=" * 20)
print("STAGE 5: Wilcoxon Test + Final Output")
print("=" * 20)
 
df = pd.read_parquet("../data/processed/papers_with_residuals.parquet")
gems = pd.read_parquet("../data/processed/hidden_gems_candidates.parquet")
 
df["paper_id"] = df["paper_id"].astype(str)
gems["paper_id"] = gems["paper_id"].astype(str)
 
print(f"All papers:     {len(df):,}")
print(f"Bridge papers:  {df['is_bridge'].sum():,}")
print(f"Hidden gems:    {len(gems):,}")

# WILCOXON RANK-SUM TEST (MANN-WHITNEY U)
# We use one-tailed because our hypothesis has a specific direction —
# we predict bridge papers are under-cited, not just different.

print("\n" + "=" * 20)
print("WILCOXON RANK-SUM TEST (Mann-Whitney U)")
print("=" * 20)
 
bridge_residuals    = df[df["is_bridge"]]["residual"].values
nonbridge_residuals = df[~df["is_bridge"]]["residual"].values
 
print(f"Bridge group:     n={len(bridge_residuals):,}")
print(f"Non-bridge group: n={len(nonbridge_residuals):,}")
print(f"\nBridge residuals:")
print(f"  Mean:   {bridge_residuals.mean():.4f}")
print(f"  Median: {np.median(bridge_residuals):.4f}")
print(f"  Std:    {bridge_residuals.std():.4f}")
print(f"\nNon-bridge residuals:")
print(f"  Mean:   {nonbridge_residuals.mean():.4f}")
print(f"  Median: {np.median(nonbridge_residuals):.4f}")
print(f"  Std:    {nonbridge_residuals.std():.4f}")
 
# Run the test
stat, p_value = stats.mannwhitneyu(
    bridge_residuals,
    nonbridge_residuals,
    alternative='less'    # test if bridge < non-bridge
)
 
print(f"\nMann-Whitney U statistic: {stat:.2f}")
print(f"p-value (one-tailed):     {p_value:.6f}")
 
if p_value < 0.001:
    significance = "highly significant (p < 0.001)"
elif p_value < 0.01:
    significance = "significant (p < 0.01)"
elif p_value < 0.05:
    significance = "significant (p < 0.05)"
else:
    significance = "NOT significant (p >= 0.05)"
 
print(f"Result: {significance}")
 
if p_value < 0.05:
    print("\n→ HYPOTHESIS CONFIRMED:")
    print("  Bridge papers are statistically significantly more")
    print("  under-cited than non-bridge papers.")
    print("  We reject the null hypothesis.")
else:
    print("\n→ HYPOTHESIS NOT CONFIRMED at p < 0.05")
    print("  The difference may be due to random chance.")
    print("  Consider dataset size or threshold adjustments.")


# EFFECT SIZE — CLIFF'S DELTA
n1 = len(bridge_residuals)
n2 = len(nonbridge_residuals)
cliffs_delta = 1 - (2 * stat) / (n1 * n2)
 
print(f"\nEffect size — Cliff's delta: {cliffs_delta:.4f}")
 
if abs(cliffs_delta) >= 0.474:
    effect_label = "large"
elif abs(cliffs_delta) >= 0.330:
    effect_label = "medium"
elif abs(cliffs_delta) >= 0.147:
    effect_label = "small"
else:
    effect_label = "negligible"
 
print(f"Effect size magnitude: {effect_label}")
print(f"Interpretation: a randomly chosen bridge paper has a")
print(f"{abs(cliffs_delta)*100:.1f}% probability of being more under-cited")
print(f"than a randomly chosen non-bridge paper.")


# BASELINE COMPARISON
print("\n" + "=" * 20)
print("BASELINE COMPARISON")
print("=" * 20)
 
n_bridge = df["is_bridge"].sum()
 
# Baseline 1: random selection of same size as bridge group
# Expected bridge papers by chance = n_bridge / total * n_bridge
expected_by_chance = (n_bridge / len(df)) * n_bridge
print(f"Baseline 1 — Random selection of {n_bridge} papers:")
print(f"  Expected bridge papers by chance: {expected_by_chance:.1f}")
print(f"  Our method found:                 {n_bridge} bridge papers")
print(f"  Enrichment factor:                {n_bridge / expected_by_chance:.1f}x")
 
# Baseline 2: lowest raw citation count papers (naive under-citation detection)
bottom_by_citations = df.nsmallest(n_bridge, "num_citations")
bridge_in_bottom = bottom_by_citations["is_bridge"].sum()
print(f"\nBaseline 2 — Bottom {n_bridge} papers by raw citation count:")
print(f"  Bridge papers in this group: {bridge_in_bottom}")
print(f"  Our bridge group size:       {n_bridge}")
pct_bottom = bridge_in_bottom / n_bridge * 100
print(f"  Overlap with our method:     {pct_bottom:.1f}%")
print(f"  → Raw citation count {'agrees' if pct_bottom > 50 else 'disagrees'} "
      f"with our bridge detection")
 
# Baseline 3: highest age papers (oldest papers should have most citations)
top_by_age = df.nlargest(n_bridge, "age")
bridge_in_old = top_by_age["is_bridge"].sum()
print(f"\nBaseline 3 — Top {n_bridge} oldest papers:")
print(f"  Bridge papers among oldest: {bridge_in_old}")
print(f"  Random expectation:         {expected_by_chance:.1f}")
if bridge_in_old < expected_by_chance:
    print(f"  → Bridge papers tend to be NEWER, not older papers")
else:
    print(f"  → Bridge papers trend older")

# FINAL HIDDEN GEMS RANKING
print("\n" + "=" * 20)
print("FINAL HIDDEN GEMS RANKING")
print("=" * 20)
 
# Final score: semantic bridge confirmation × magnitude of under-citation
# abs(residual) because residual is negative for under-cited papers
# We want large magnitude (very under-cited) AND high semantic score
gems["final_score"] = (
    gems["bridge_semantic_score"] * gems["residual"].abs()
)
 
# Rank by final score
gems_ranked = gems.sort_values("final_score", ascending=False).reset_index(drop=True)
gems_ranked["rank"] = gems_ranked.index + 1
 
# Which two communities does each paper bridge?
# home community + top neighbor community
gems_ranked["bridges_from"] = gems_ranked["community_hierarchical"]
gems_ranked["bridges_to"]   = gems_ranked["top_neighbor_comm"]
 
print(f"Final hidden gems: {len(gems_ranked)}")
print(f"\nTop 15 Hidden Gems:")
 
display_cols = ["rank", "title", "year", "real_in_degree",
                "bridge_score", "residual", "bridge_semantic_score",
                "final_score", "bridges_from", "bridges_to"]
 
print(gems_ranked[display_cols].head(15).to_string(index=False))

# VISUALIZATION 1 — RESIDUAL DISTRIBUTION
print("\nGenerating residual distribution plot...")
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# Plot 1: Residual distributions for bridge vs non-bridge
ax1 = axes[0]
bins = np.linspace(-3, 4, 60)
ax1.hist(nonbridge_residuals, bins=bins, alpha=0.5,
         color="steelblue", label=f"Non-bridge (n={len(nonbridge_residuals):,})",
         density=True)
ax1.hist(bridge_residuals, bins=bins, alpha=0.7,
         color="crimson", label=f"Bridge (n={len(bridge_residuals):,})",
         density=True)
ax1.axvline(x=np.median(nonbridge_residuals), color="steelblue",
            linestyle="--", linewidth=2, label=f"Non-bridge median: "
            f"{np.median(nonbridge_residuals):.3f}")
ax1.axvline(x=np.median(bridge_residuals), color="crimson",
            linestyle="--", linewidth=2, label=f"Bridge median: "
            f"{np.median(bridge_residuals):.3f}")
ax1.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
ax1.set_xlabel("Residual (actual - predicted log citations)")
ax1.set_ylabel("Density")
ax1.set_title(f"Citation Residuals: Bridge vs Non-Bridge\n"
              f"p={p_value:.4f}, Cliff's δ={cliffs_delta:.4f} ({effect_label})")
ax1.legend(fontsize=8)
 
# Plot 2: Top hidden gems — how under-cited are they?
ax2 = axes[1]
top15_gems = gems_ranked.head(15)
colors = ["gold" if r == 1 else "darkorange" if r <= 5 else "coral"
          for r in top15_gems["rank"]]
bars = ax2.barh(
    range(len(top15_gems)),
    top15_gems["residual"].abs(),
    color=colors
)
ax2.set_yticks(range(len(top15_gems)))
ax2.set_yticklabels(
    [t[:45] + "..." if len(t) > 45 else t
     for t in top15_gems["title"]],
    fontsize=7
)
ax2.invert_yaxis()
ax2.set_xlabel("Magnitude of under-citation |residual|")
ax2.set_title("Top 15 Hidden Gems\n(longer bar = more under-cited)")
 
gold_patch  = mpatches.Patch(color="gold",       label="Rank 1")
orange_patch = mpatches.Patch(color="darkorange", label="Ranks 2-5")
coral_patch = mpatches.Patch(color="coral",       label="Ranks 6-15")
ax2.legend(handles=[gold_patch, orange_patch, coral_patch], fontsize=8)
 
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "residual_distribution.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {plot_path}")


# VISUALIZATION 2 — NETWORK SUBGRAPH
print("\nGenerating network visualization...")
 
try:
    from pyvis.network import Network
 
    # Load graph
    with open("../data/processed/citation_graph.pkl", "rb") as f:
        G = pickle.load(f)
 
    gem_ids = set(gems_ranked["paper_id"].tolist())
 
    # Build subgraph: gems + their direct neighbors (real papers only)
    real_paper_ids = set(df["paper_id"].tolist())
    subgraph_nodes = set(gem_ids)
 
    for gid in gem_ids:
        if gid in G:
            for neighbor in list(G.predecessors(gid)) + list(G.successors(gid)):
                if str(neighbor) in real_paper_ids:
                    subgraph_nodes.add(str(neighbor))
 
    # Limit to manageable size
    if len(subgraph_nodes) > 300:
        # Keep gems + top neighbors by degree
        neighbor_ids = subgraph_nodes - gem_ids
        neighbor_degrees = {
            n: G.degree(n) for n in neighbor_ids if n in G
        }
        top_neighbors = sorted(
            neighbor_degrees.items(), key=lambda x: x[1], reverse=True
        )[:200]
        subgraph_nodes = gem_ids | {n for n, _ in top_neighbors}
 
    print(f"  Subgraph: {len(subgraph_nodes)} nodes")
 
    # Build paper metadata lookup
    paper_meta = df.set_index("paper_id")[
        ["title", "community_hierarchical", "is_bridge", "num_citations"]
    ].to_dict("index")
 
    # Create pyvis network
    net = Network(
        height="750px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True
    )
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "springLength": 100
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 150}
      },
      "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
                "color": {"opacity": 0.4}},
      "interaction": {"hover": true, "tooltipDelay": 100}
    }
    """)
 
    # Color map for communities
    import hashlib
    def community_color(comm):
        if comm is None:
            return "#888888"
        h = int(hashlib.md5(str(comm).encode()).hexdigest()[:6], 16)
        r = (h >> 16) & 0xFF
        g = (h >> 8) & 0xFF
        b = h & 0xFF
        # Ensure not too dark
        r = max(r, 80)
        g = max(g, 80)
        b = max(b, 80)
        return f"#{r:02x}{g:02x}{b:02x}"
 
    # Add nodes
    for node_id in subgraph_nodes:
        meta = paper_meta.get(str(node_id), {})
        title_text = str(meta.get("title", node_id))[:80]
        comm = meta.get("community_hierarchical", None)
        citations = meta.get("num_citations", 0)
        is_gem = str(node_id) in gem_ids
 
        if is_gem:
            gem_row = gems_ranked[gems_ranked["paper_id"] == str(node_id)]
            rank = int(gem_row["rank"].values[0]) if len(gem_row) > 0 else 99
            color  = "#FFD700"   # gold
            size   = 30
            border = "#FF6600"
            label  = f"#{rank}: {title_text[:30]}..."
            tooltip = (f"HIDDEN GEM #{rank}\n"
                      f"{title_text}\n"
                      f"Community: {comm}\n"
                      f"Citations: {citations}\n"
                      f"Bridge score: {gem_row['bridge_score'].values[0]:.4f}\n"
                      f"Residual: {gem_row['residual'].values[0]:.4f}")
        else:
            color  = community_color(comm)
            size   = 10
            border = color
            label  = ""
            tooltip = f"{title_text}\nCommunity: {comm}\nCitations: {citations}"
 
        net.add_node(
            str(node_id),
            label=label,
            title=tooltip,
            color={"background": color, "border": border},
            size=size,
        )
 
    # Add edges within subgraph
    edge_count = 0
    for node_id in subgraph_nodes:
        if node_id not in G:
            continue
        for successor in G.successors(node_id):
            if str(successor) in subgraph_nodes:
                net.add_edge(
                    str(node_id), str(successor),
                    color="#444477",
                    width=1
                )
                edge_count += 1
 
    print(f"  Edges in subgraph: {edge_count}")
 
    html_path = os.path.join(OUTPUT_DIR, "network_visualization.html")
    net.save_graph(html_path)
    print(f"  Saved → {html_path}")
    print("  Open this file in your browser to explore interactively.")
 
except ImportError:
    print("  pyvis not installed. Skipping interactive visualization.")
    print("  Run: pip install pyvis")
except Exception as e:
    print(f"  Visualization error: {e}")
    print("  Skipping network visualization.")


# SAVE FINAL OUTPUT
print("\n" + "=" * 20)
print("Saving final outputs")
print("=" * 20)
 
# Final CSV — the main deliverable
final_cols = [
    "rank", "paper_id", "title", "year", "doi",
    "real_in_degree", "num_citations",
    "bridge_score", "residual",
    "bridge_semantic_score", "final_score",
    "bridges_from", "bridges_to",
    "community_hierarchical",
    "sim_home_community", "sim_top_neighbor"
]
 
# Add doi from main df
gems_ranked = gems_ranked.merge(
    df[["paper_id", "doi", "num_citations"]],
    on="paper_id",
    how="left",
    suffixes=("", "_df")
)
 
# Use the merged num_citations if original is missing
if "num_citations_df" in gems_ranked.columns:
    gems_ranked["num_citations"] = gems_ranked["num_citations"].fillna(
        gems_ranked["num_citations_df"]
    )
    gems_ranked.drop(columns=["num_citations_df"], inplace=True)
 
# Select only columns that exist
available_cols = [c for c in final_cols if c in gems_ranked.columns]
final_df = gems_ranked[available_cols].copy()
 
csv_path = os.path.join(OUTPUT_DIR, "hidden_gems_final.csv")
final_df.to_csv(csv_path, index=False)
print(f"Final hidden gems CSV → {csv_path}")
 
# Wilcoxon stats report
wilcoxon_stats = f"""
=== Wilcoxon Statistical Test Results ===
 
HYPOTHESIS
  H0: Bridge papers and non-bridge papers have the same
      citation residual distribution.
  H1: Bridge papers have more negative residuals (under-cited).
 
GROUPS
  Bridge papers:     n={len(bridge_residuals):,}
  Non-bridge papers: n={len(nonbridge_residuals):,}
 
DESCRIPTIVE STATISTICS
  Bridge residuals:
    Mean:   {bridge_residuals.mean():.4f}
    Median: {np.median(bridge_residuals):.4f}
    Std:    {bridge_residuals.std():.4f}
  Non-bridge residuals:
    Mean:   {nonbridge_residuals.mean():.4f}
    Median: {np.median(nonbridge_residuals):.4f}
    Std:    {nonbridge_residuals.std():.4f}
 
TEST RESULTS
  Test:           Mann-Whitney U (one-tailed, alternative='less')
  U statistic:    {stat:.2f}
  p-value:        {p_value:.6f}
  Result:         {significance}
 
EFFECT SIZE
  Cliff's delta:  {cliffs_delta:.4f}
  Magnitude:      {effect_label}
  Interpretation: A randomly chosen bridge paper has a
                  {abs(cliffs_delta)*100:.1f}% probability of being
                  more under-cited than a non-bridge paper.
 
BASELINE COMPARISONS
  Random selection enrichment:   {n_bridge / expected_by_chance:.1f}x
  Bottom citation overlap:       {bridge_in_bottom}/{n_bridge} papers ({pct_bottom:.1f}%)
  Bridge papers among oldest:    {bridge_in_old}
 
HIDDEN GEMS
  Total identified:              {len(gems_ranked)}
  (is_top_bridge + semantically validated + under-cited)
 
TOP 10 HIDDEN GEMS:
{gems_ranked[['rank','title','year','bridge_score',
              'residual','final_score']].head(10).to_string(index=False)}
"""
 
wilcoxon_path = os.path.join(OUTPUT_DIR, "wilcoxon_stats.txt")
with open(wilcoxon_path, "w") as f:
    f.write(wilcoxon_stats)
print(f"Wilcoxon stats → {wilcoxon_path}")
 
print("\n" + "=" * 20)
print("PIPELINE COMPLETE")
print("=" * 20)
print(f"""
Summary:
  Papers analyzed:         {len(df):,}
  Communities detected:    {df['community_hierarchical'].nunique():,}
  Bridge papers found:     {df['is_bridge'].sum():,}
  Top bridge candidates:   {df['is_top_bridge'].sum():,}
  Semantically validated:  71
  Hidden gems (final):     {len(gems_ranked)}
 
Statistical result:        {significance}
Effect size:               Cliff's delta = {cliffs_delta:.4f} ({effect_label})
 
Final deliverables:
  hidden_gems_final.csv          — ranked list of hidden gems
  wilcoxon_stats.txt             — statistical proof
  residual_distribution.png      — visualization 1
  network_visualization.html     — visualization 2 (open in browser)
""")

# Add to your wilcoxon script — compare top 10% bridges vs non-bridge
top_bridge_residuals = df[df["is_top_bridge"] == True]["residual"].values
stat2, p2 = stats.mannwhitneyu(
    top_bridge_residuals,
    nonbridge_residuals,
    alternative='less'
)
print(f"Top bridge papers (n={len(top_bridge_residuals)}) vs non-bridge:")
print(f"  p-value: {p2:.4f}")
print(f"  Top bridge mean residual: {top_bridge_residuals.mean():.4f}")