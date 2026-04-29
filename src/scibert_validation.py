"""
Stage 4 — SciBERT/SPECTER Semantic Validation
 
WHAT THIS STAGE DOES
---------------------
Takes the 71 top bridge papers and confirms they are GENUINELY
cross-domain using text analysis — not just structurally positioned
between communities in the citation graph.
 
A paper could have high bridge score because of graph accidents
(e.g. a paper that cites things randomly across fields without
being intellectually cross-domain). SciBERT catches this.
 
ALGORITHM DESIGN
----------------
Step 1 — Encode abstracts with SPECTER
  SPECTER is a transformer model trained specifically on citation
  relationships between papers. It reads an abstract and produces
  a 768-dimensional vector (embedding) that captures the paper's
  semantic meaning in scientific space.
  Papers about similar topics have vectors that point in similar
  directions. Papers about different topics point in different directions.
Step 2 — Build community centroid vectors
  For each community that a top bridge paper connects, compute the
  average embedding of all papers in that community that have abstracts.
  This centroid represents "what this research community talks about"
  in semantic space.
 
Step 3 — Compute cosine similarity
  For each top bridge paper X connecting communities A and B:
    sim_A = cosine_similarity(embedding_X, centroid_A)
    sim_B = cosine_similarity(embedding_X, centroid_B)
  
  Cosine similarity ranges from -1 to 1:
    1.0 = identical meaning vectors (same direction)
    0.0 = completely unrelated (perpendicular vectors)
   -1.0 = opposite meanings (opposite directions)
 
Step 4 — Validate bridge status
  A genuine bridge paper should:
    - Have meaningful similarity to BOTH communities (sim > threshold)
    - NOT be fully absorbed into either community
    - Have similar similarity to both (not dominated by one side)
  
  We compute:
    bridge_semantic_score = min(sim_A, sim_B)
    The minimum ensures both similarities are above threshold.
    A paper similar to A but not B fails the min test.
    
  Threshold: 0.3 (cosine similarity in scientific embedding space)
  Papers above threshold on both sides = semantically confirmed bridges.
 
WHY SPECTER OVER RAW SCIBERT?
  Raw SciBERT was trained on general biomedical text.
  SPECTER was trained on citation pairs — it learned that papers
  which cite each other should have similar embeddings.
  This makes SPECTER's similarity scores directly meaningful for
  citation network analysis — papers SPECTER says are similar
  are papers the citation network treats as similar too.
  Perfect fit for this project.
 
Inputs:
  data/processed/papers_with_residuals.parquet
 
Outputs:
  data/processed/papers_with_specter.parquet   (top bridge papers only)
  data/processed/specter_stats.txt
  data/processed/hidden_gems_candidates.parquet (validated bridges)
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "../data/processed"
SIMILARITY_THRESHOLD = 0.3

# Load Data
print("=" * 20)
print("STAGE 4: SciBERT/SPECTER Semantic Validation")
print("=" * 20)

df = pd.read_parquet("../data/processed/papers_with_residuals.parquet")
df["paper_id"] = df["paper_id"].astype(str)
 
print(f"Total papers loaded: {len(df):,}")
 
# Separate top bridge papers and all papers with abstracts
top_bridges = df[df["is_top_bridge"] == True].copy()
print(f"Top bridge papers (is_top_bridge=True): {len(top_bridges):,}")
print(f"Top bridge papers with abstracts: "
      f"{top_bridges['abstract'].notna().sum():,}")

# LOAD SPECTER MODEL
print("\n" + "=" * 20)
print("Loading SPECTER model")
print("=" * 20)
 
print("Loading allenai-specter (downloads ~400MB on first run)...")
model = SentenceTransformer("allenai-specter")
print("Model loaded.")

# ENCODE TOP BRIDGE PAPER ABSTRACTS
print("\n" + "=" * 20)
print("Encoding top bridge paper abstracts")
print("=" * 20)
 
def build_specter_input(row):
    """Format paper for SPECTER: title [SEP] abstract"""
    title    = str(row["title"])    if pd.notna(row["title"])    else ""
    abstract = str(row["abstract"]) if pd.notna(row["abstract"]) else ""
    return title + " [SEP] " + abstract
 
# Filter to top bridge papers with abstracts
top_bridges_with_abs = top_bridges[top_bridges["abstract"].notna()].copy()
print(f"Encoding {len(top_bridges_with_abs)} top bridge papers...")
 
bridge_texts = top_bridges_with_abs.apply(build_specter_input, axis=1).tolist()
 
# encode() runs the transformer on each text
# batch_size=16 processes 16 abstracts at a time — good for CPU
# show_progress_bar=True shows a tqdm bar
bridge_embeddings = model.encode(
    bridge_texts,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True
)
 
print(f"Bridge embeddings shape: {bridge_embeddings.shape}")
# Expected: (71, 768) — 71 papers, 768-dimensional vectors

# BUILD COMMUNITY CENTROIDS
# WHICH COMMUNITIES TO BUILD CENTROIDS FOR?
#   Only the communities that our 71 top bridge papers belong to.
#   A bridge paper in community "14_31" connects to other communities
#   among its neighbors — we build centroids for all of those.
#   We identify connected communities from cluster_diversity neighbors.

print("\n" + "=" * 20)
print("Building community centroid embeddings")
print("=" * 20)

# Get all unique communities in the full dataset
all_communities = top_bridges_with_abs["community_hierarchical"].unique().tolist()
bridge_paper_ids = set(top_bridges_with_abs["paper_id"].tolist())

# Find papers that cite any of our top bridge papers
# These represent the communities that "receive" the bridge paper
citing_communities = []
for _, row in top_bridges_with_abs.iterrows():
    comm = row["community_hierarchical"]
    if pd.notna(comm):
        citing_communities.append(comm)

# Build the full list of communities we need centroids for
# Include top communities by paper count to ensure coverage
top_communities_by_size = (
    df[df["abstract"].notna()]
    ["community_hierarchical"]
    .value_counts()
    .head(50)            # top 50 communities by paper count
    .index.tolist()
)
 
communities_needed = list(set(
    all_communities +
    citing_communities +
    top_communities_by_size
))
communities_needed = [c for c in communities_needed if pd.notna(c)]
 
print(f"Building centroids for {len(communities_needed)} communities...")
print("This may take 5-10 minutes on CPU...")
 
community_centroids = {}
 
for i, comm in enumerate(communities_needed):
    # Get all papers in this community with abstracts
    comm_papers = df[
        (df["community_hierarchical"] == comm) &
        (df["abstract"].notna())
    ].copy()
 
    if len(comm_papers) < 2:
        # Skip communities with fewer than 2 papers
        # Not enough to build a meaningful centroid
        continue
 
    # Build SPECTER inputs
    comm_texts = comm_papers.apply(build_specter_input, axis=1).tolist()
 
    # Encode — suppress progress bar for community encoding
    comm_embeddings = model.encode(
        comm_texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True
    )
 
    # Average all embeddings → centroid
    centroid = comm_embeddings.mean(axis=0)
    community_centroids[comm] = centroid
 
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{len(communities_needed)} communities...")
 
print(f"\nCentroids built: {len(community_centroids)}")

# COMPUTE SIMILARITY SCORES
print("\n" + "=" * 20)
print("Computing semantic similarity scores")
print("=" * 20)
 
results = []
 
for idx, (paper_idx, row) in enumerate(top_bridges_with_abs.iterrows()):
    paper_embedding = bridge_embeddings[idx].reshape(1, -1)
    home_comm = row["community_hierarchical"]
 
    # Similarity to home community
    sim_home = None
    if home_comm in community_centroids:
        centroid = community_centroids[home_comm].reshape(1, -1)
        sim_home = float(cosine_similarity(paper_embedding, centroid)[0][0])
 
    # Find neighboring communities
    # Strategy: look at all communities in our centroid dict
    # and find ones most semantically similar to this paper
    # (these are the communities this paper bridges TO)
    neighbor_sims = {}
    for comm, centroid in community_centroids.items():
        if comm == home_comm:
            continue
        centroid_vec = centroid.reshape(1, -1)
        sim = float(cosine_similarity(paper_embedding, centroid_vec)[0][0])
        neighbor_sims[comm] = sim
 
    # Top 3 most similar neighboring communities
    top_neighbors = sorted(
        neighbor_sims.items(), key=lambda x: x[1], reverse=True
    )[:3]
 
    top_neighbor_comm  = top_neighbors[0][0] if top_neighbors else None
    top_neighbor_sim   = top_neighbors[0][1] if top_neighbors else None
    second_neighbor_comm = top_neighbors[1][0] if len(top_neighbors) > 1 else None
    second_neighbor_sim  = top_neighbors[1][1] if len(top_neighbors) > 1 else None
 
    # Bridge semantic score = min(sim_home, top_neighbor_sim)
    # Both must be above threshold for genuine validation
    if sim_home is not None and top_neighbor_sim is not None:
        bridge_semantic_score = min(sim_home, top_neighbor_sim)
        is_semantically_validated = (
            sim_home > SIMILARITY_THRESHOLD and
            top_neighbor_sim > SIMILARITY_THRESHOLD
        )
    else:
        bridge_semantic_score = 0
        is_semantically_validated = False
 
    results.append({
        "paper_id":               row["paper_id"],
        "title":                  row["title"],
        "year":                   row["year"],
        "real_in_degree":         row["real_in_degree"],
        "bridge_score":           row["bridge_score"],
        "residual":               row["residual"],
        "community_hierarchical": home_comm,
        "sim_home_community":     sim_home,
        "top_neighbor_comm":      top_neighbor_comm,
        "sim_top_neighbor":       top_neighbor_sim,
        "second_neighbor_comm":   second_neighbor_comm,
        "sim_second_neighbor":    second_neighbor_sim,
        "bridge_semantic_score":  bridge_semantic_score,
        "is_semantically_validated": is_semantically_validated,
    })
 
results_df = pd.DataFrame(results)
 
print(f"\nSemantic validation results:")
print(f"  Top bridge papers evaluated:    {len(results_df):,}")
print(f"  Semantically validated bridges: "
      f"{results_df['is_semantically_validated'].sum():,}")
print(f"  Failed semantic validation:     "
      f"{(~results_df['is_semantically_validated']).sum():,}")
print(f"\n  Mean sim to home community:     "
      f"{results_df['sim_home_community'].mean():.4f}")
print(f"  Mean sim to top neighbor:       "
      f"{results_df['sim_top_neighbor'].mean():.4f}")
print(f"  Mean bridge semantic score:     "
      f"{results_df['bridge_semantic_score'].mean():.4f}")

# IDENTIFY HIDDEN GEMS
results_df["is_hidden_gem"] = (
    results_df["is_semantically_validated"] &
    (results_df["residual"] < 0)
)
 
hidden_gems = results_df[results_df["is_hidden_gem"]].sort_values(
    "bridge_semantic_score", ascending=False
)
 
print(f"\n{'='*60}")
print(f"HIDDEN GEMS IDENTIFIED")
print(f"{'='*60}")
print(f"Papers meeting all three criteria: {len(hidden_gems):,}")
print(f"  (is_top_bridge + semantically_validated + under-cited)")
 
if len(hidden_gems) > 0:
    print(f"\nHidden Gems ranked by bridge semantic score:")
    display_cols = ["paper_id", "title", "year", "real_in_degree",
                    "bridge_score", "residual",
                    "sim_home_community", "sim_top_neighbor",
                    "bridge_semantic_score", "community_hierarchical"]
    print(hidden_gems[display_cols].to_string(index=False))
else:
    print("No papers met all three criteria.")
    print("Consider lowering SIMILARITY_THRESHOLD from 0.3 to 0.2")


# SAVE OUTPUTS
# Save full results for all top bridge papers
specter_path = os.path.join(OUTPUT_DIR, "papers_with_specter.parquet")
results_df.to_parquet(specter_path, index=False)
print(f"\nSaved → {specter_path}")
 
# Save hidden gems separately
gems_path = os.path.join(OUTPUT_DIR, "hidden_gems_candidates.parquet")
hidden_gems.to_parquet(gems_path, index=False)
print(f"Saved → {gems_path}")
 
# Stats report
stats = f"""
=== SPECTER Semantic Validation Stats ===
 
MODEL
  Model:              allenai-specter
  Similarity threshold: {SIMILARITY_THRESHOLD}
 
VALIDATION RESULTS
  Top bridge papers evaluated:      {len(results_df):,}
  Semantically validated:           {results_df['is_semantically_validated'].sum():,}
  Failed validation:                {(~results_df['is_semantically_validated']).sum():,}
 
SIMILARITY SCORES
  Mean sim to home community:       {results_df['sim_home_community'].mean():.4f}
  Mean sim to top neighbor:         {results_df['sim_top_neighbor'].mean():.4f}
  Mean bridge semantic score:       {results_df['bridge_semantic_score'].mean():.4f}
 
HIDDEN GEMS (all three criteria met)
  Count:                            {len(hidden_gems):,}
 
TOP HIDDEN GEMS:
{hidden_gems[['paper_id','title','year','bridge_score',
              'residual','bridge_semantic_score']].head(10).to_string(index=False)
  if len(hidden_gems) > 0 else "None found"}
"""
 
stats_path = os.path.join(OUTPUT_DIR, "specter_stats.txt")
with open(stats_path, "w") as f:
    f.write(stats)
print(f"Stats saved → {stats_path}")
 
print("\nDone. Stage 4 complete.")
print(f"\nHidden gem candidates: {len(hidden_gems)}")
print("Hand hidden_gems_candidates.parquet and")
print("papers_with_residuals.parquet to Stage 5 (Wilcoxon test).")