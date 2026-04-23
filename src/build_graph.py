"""
build_graph.py
WS1 — Graph construction from cleaned PubMed JSON data
Branch: tanveer-imp

Outputs:
  - data/processed/citation_graph.pkl   (NetworkX DiGraph)
  - data/processed/papers.parquet       (enriched paper table)
  - data/processed/graph_stats.txt      (sanity check report)
"""

import json
import pickle
import os
import networkx as nx
import pandas as pd
from datetime import datetime
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────

INPUT_PATH = "../data/raw/train_data.jsonl"          # update if your filename differs
OUTPUT_DIR = "../data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Load JSONL ──────────────────────────────────────────────────────────────

print("Loading JSONL...")
records = []
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"  Loaded {len(records):,} records")


# ── 2. Parse and clean records ────────────────────────────────────────────────

def parse_year(pub_date):
    """
    pubDate comes in as a Unix timestamp in milliseconds.
    e.g. 1177977600000 → 2007
    Falls back to None if unparseable.
    """
    if pub_date is None:
        return None
    try:
        ts = int(pub_date) / 1000          # ms → seconds
        return datetime.utcfromtimestamp(ts).year
    except Exception:
        return None


def clean_citations(citations):
    """
    Deduplicate citation IDs and return as a list of strings.
    Input is already a list from the cleaned JSON.
    """
    if not citations:
        return []
    return list(dict.fromkeys(str(c) for c in citations))   # preserves order, removes dupes


rows = []
skipped = 0

for rec in tqdm(records, desc="Parsing records"):
    pid = str(rec.get("publication_ID", "")).strip()
    if not pid:
        skipped += 1
        continue

    abstract = rec.get("abstract", "") or ""
    if len(abstract.strip()) < 30:       # too short to be useful for SciBERT
        abstract = None

    rows.append({
        "paper_id":       pid,
        "year":           parse_year(rec.get("pubDate")),
        "title":          rec.get("title", ""),
        "journal":        rec.get("journal", ""),
        "abstract":       abstract,
        "keywords":       rec.get("keywords", []),
        "num_citations":  rec.get("num_citations", 0),   # in-degree as reported
        "doi":            rec.get("doi", ""),
        "citing_ids":     clean_citations(rec.get("Citations", [])),
    })

print(f"  Parsed: {len(rows):,}  |  Skipped (no ID): {skipped}")


# ── 3. Build the DataFrame ────────────────────────────────────────────────────

df = pd.DataFrame(rows)

# All valid paper IDs in the dataset — used to filter edges to known nodes only
known_ids = set(df["paper_id"].tolist())

print(f"\nDataFrame shape: {df.shape}")
print(f"Papers with abstracts: {df['abstract'].notna().sum():,}")
print(f"Papers missing abstracts: {df['abstract'].isna().sum():,}")
print(f"Year range: {df['year'].min()} – {df['year'].max()}")


# ── 4. Build the NetworkX DiGraph ─────────────────────────────────────────────

print("\nBuilding citation graph...")
G = nx.DiGraph()

# Add all nodes first with metadata attached
for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding nodes"):
    G.add_node(
        row["paper_id"],
        year=row["year"],
        title=row["title"],
        journal=row["journal"],
        num_citations=row["num_citations"],
        has_abstract=row["abstract"] is not None,
    )

# Add directed edges: paper → paper it cites
# Only add edges where BOTH ends are known nodes (internal citations only)
edge_count = 0
skipped_edges = 0

for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding edges"):
    src = row["paper_id"]
    for tgt in row["citing_ids"]:
        G.add_edge(src, tgt)

print(f"  Edges added (internal):  {edge_count:,}")
print(f"  Edges skipped (external): {skipped_edges:,}")


# ── 5. Compute actual in-degree (true citation count within this dataset) ──────

print("\nComputing in-degrees...")
in_degree_map = dict(G.in_degree())
df["in_degree"] = df["paper_id"].map(in_degree_map).fillna(0).astype(int)

# out-degree = number of references this paper makes (within dataset)
out_degree_map = dict(G.out_degree())
df["out_degree"] = df["paper_id"].map(out_degree_map).fillna(0).astype(int)


# ── 6. Sanity checks ──────────────────────────────────────────────────────────

stats = f"""
=== Graph Sanity Check ===
Nodes:              {G.number_of_nodes():,}
Edges:              {G.number_of_edges():,}
Is directed:        {G.is_directed()}
Avg out-degree:     {df['out_degree'].mean():.2f}   (references per paper)
Avg in-degree:      {df['in_degree'].mean():.2f}    (citations per paper within dataset)
Max in-degree:      {df['in_degree'].max()}          (most-cited paper)
Papers with 0 citations (in dataset): {(df['in_degree'] == 0).sum():,}
Papers with abstracts:  {df['abstract'].notna().sum():,}
Year range:         {df['year'].min()} – {df['year'].max()}

Top 5 most-cited papers:
{df.nlargest(5, 'in_degree')[['paper_id','title','in_degree','year']].to_string(index=False)}
"""

print(stats)

# Save stats to file
stats_path = os.path.join(OUTPUT_DIR, "graph_stats.txt")
with open(stats_path, "w") as f:
    f.write(stats)
print(f"Stats saved → {stats_path}")


# ── 7. Save outputs ───────────────────────────────────────────────────────────

# Save graph as pickle
graph_path = os.path.join(OUTPUT_DIR, "citation_graph.pkl")
with open(graph_path, "wb") as f:
    pickle.dump(G, f)
print(f"Graph saved → {graph_path}")

# Save enriched DataFrame as parquet
parquet_path = os.path.join(OUTPUT_DIR, "papers.parquet")
df.drop(columns=["citing_ids"]).to_parquet(parquet_path, index=False)
print(f"Papers table saved → {parquet_path}")

print("\nDone. WS2 can now load citation_graph.pkl and papers.parquet.")