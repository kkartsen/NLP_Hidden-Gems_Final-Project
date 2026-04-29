"""
citation_regression.py
Stage 3: Citation Regression Model

What this stage does:
- trains a regression model to predict how many citations each paper should have received based
on its network position and properties

The gap between predicted and actual citations = residuals
Negative residual = paper got FEWER citations than expected = under-cited.
 
We then test: do bridge papers have more negative residuals than
non-bridge papers? If yes, bridge papers are systematically under-cited.

ALGORITHM DESIGN
----------------
Why regression and not just raw citation counts?
  Raw citation count is confounded by many factors:
    - Age: a 2000 paper had 14 years to accumulate citations by 2014
            a 2013 paper had only 1 year
    - Field size: a paper in a large community has more potential citers
    - Out-degree: papers that cite more tend to be cited more in return
  Regression controls for ALL of these simultaneously.
  The residual is the citation count stripped of all confounders —
  it isolates the pure under/over-citation signal.
 
Why Gradient Boosting (LightGBM)?
  Citation counts are not linearly related to features.
  The relationship between age and citations is non-linear (accelerating).
  LightGBM handles non-linear relationships, feature interactions,
  and skewed targets naturally without manual feature engineering.
  It also gives feature importance scores which show your professor
  which features matter most for citation prediction.
 
Why log-transform the target?
  Citation counts are extremely right-skewed — a few papers have
  hundreds of citations, most have very few.
  Training on raw counts would make the model obsess over predicting
  the few high-citation papers correctly, ignoring the majority.
  log(1 + citations) compresses the scale — a paper going from
  0 to 1 citation is as important as one going from 10 to 27.
  We transform back to original scale for interpretation.
 
Inputs:
  data/processed/papers_with_bridges.parquet
 
Outputs:
  data/processed/papers_with_residuals.parquet  (main output for Stage 5)
  data/processed/regression_stats.txt
  data/processed/feature_importance.png
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "../data/processed"

# Load data
print("=" * 20)
print("STAGE 3: Citation Regression Model")
print("=" * 20)

df = pd.read_parquet("../data/processed/papers_with_bridges.parquet")
df["paper_id"] = df["paper_id"].astype(str)
 
print(f"Papers loaded: {len(df):,}")
print(f"Columns: {list(df.columns)}")

# Feature Engineering
print("\n" + "=" * 20)
print("Feature Engineering")
print("=" * 20)

DATASET_END_YEAR = 2014
 
# Age of paper in years
df["age"] = DATASET_END_YEAR - df["year"].fillna(DATASET_END_YEAR)
df["age"] = df["age"].clip(lower=0)   # no negative ages
 
# Community size — how many papers share this paper's community
comm_size_map = df["community_hierarchical"].value_counts().to_dict()
df["community_size"] = df["community_hierarchical"].map(comm_size_map).fillna(1)
 
# Encode community as integer for LightGBM
# LightGBM can use categorical features but needs integer encoding
le = LabelEncoder()
df["community_encoded"] = le.fit_transform(
    df["community_hierarchical"].fillna("unknown")
)
 
# Fill any remaining NaN values in features
feature_cols = [
    "age",                      # time available to accumulate citations
    "cluster_diversity",        # structural position diversity
    "bridge_score",             # bridge signal
    "citation_community_count", # cross-community recognition
    "community_size",           # size of home community
]
 
for col in feature_cols:
    df[col] = df[col].fillna(0)
 
# Target variable
TARGET = "num_citations"
df[TARGET] = df[TARGET].fillna(0).clip(lower=0)
 
print(f"Features: {feature_cols}")
print(f"Target: {TARGET}")
print(f"\nTarget distribution:")
print(f"  Mean citations:    {df[TARGET].mean():.2f}")
print(f"  Median citations:  {df[TARGET].median():.2f}")
print(f"  Max citations:     {df[TARGET].max()}")
print(f"  Papers with 0:     {(df[TARGET] == 0).sum():,}")
print(f"  Papers with 1-5:   {((df[TARGET] >= 1) & (df[TARGET] <= 5)).sum():,}")
print(f"  Papers with 6-20:  {((df[TARGET] >= 6) & (df[TARGET] <= 20)).sum():,}")
print(f"  Papers with 20+:   {(df[TARGET] > 20).sum():,}")
 
# Log transform target
# log1p(x) = log(1+x), handles 0 citations gracefully (log(1+0) = 0)
df["log_citations"] = np.log1p(df[TARGET])
 
print(f"\nAfter log transform:")
print(f"  Mean log citations:   {df['log_citations'].mean():.4f}")
print(f"  Max log citations:    {df['log_citations'].max():.4f}")

# Train / Test Split

# NOTE: We compute residuals on ALL papers (not just test set)
#   because we need residuals for every paper to run the Wilcoxon test.
#   The train/test split is only for evaluating model quality (R² score).
 
print("\n" + "=" * 60)
print("Train/Test Split")
print("=" * 60)
 
X = df[feature_cols]
y = df["log_citations"]
 
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index,
    test_size=0.2,
    random_state=42
)
 
print(f"Training set:   {len(X_train):,} papers")
print(f"Test set:       {len(X_test):,} papers")

# Train Lightgbm model 
# LIGHTGBM HYPERPARAMETERS EXPLAINED:
#
# n_estimators=500
#   Number of decision trees to build. More trees = better fit
#   but slower. 500 is a good balance for this dataset size.
#
# learning_rate=0.05
#   How much each tree corrects the previous trees' errors.
#   Lower = more trees needed but more stable/accurate.
#   0.05 is standard for LightGBM.
#
# num_leaves=31
#   Controls tree complexity. More leaves = can fit more complex
#   patterns but risks overfitting. 31 is LightGBM's default.
#
# min_child_samples=20
#   Minimum papers in a leaf node. Prevents overfitting on
#   tiny subgroups. Important for sparse features like bridge_score.
#
# subsample=0.8
#   Each tree uses 80% of training data randomly selected.
#   Reduces overfitting, adds diversity to the ensemble.
#
# colsample_bytree=0.8
#   Each tree uses 80% of features randomly selected.
#   Same purpose as subsample.
#
# reg_alpha=0.1, reg_lambda=0.1
#   L1 and L2 regularization. Penalizes large weights,
#   prevents the model from overfitting to noisy features.
 
print("\n" + "=" * 60)
print("Training LightGBM Model")
print("=" * 60)
 
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1          # suppress LightGBM training output
)
 
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False),
               lgb.log_evaluation(100)]
)
 
print(f"Best iteration: {model.best_iteration_}")

# EVALUATE MODEL
print("\n" + "=" * 20)
print("Model Evaluation")
print("=" * 20)
 
y_pred_test = model.predict(X_test)
 
mse  = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred_test)
 
print(f"Test set performance:")
print(f"  R² score:  {r2:.4f}  (1.0 = perfect, 0.0 = no better than mean)")
print(f"  RMSE:      {rmse:.4f}  (in log-citation units)")
 
# Interpret R²
if r2 >= 0.5:
    print(f"  → Strong predictive power. Residuals are meaningful.")
elif r2 >= 0.3:
    print(f"  → Moderate predictive power. Residuals are usable.")
elif r2 >= 0.1:
    print(f"  → Weak but nonzero predictive power.")
else:
    print(f"  → Very low R². Citations are hard to predict from these features.")
    print(f"  → This is common for citation data — still valid for Wilcoxon test.")
 
# Feature importance
importances = model.feature_importances_
feat_imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values("importance", ascending=False)
 
print(f"\nFeature importances (higher = more useful for prediction):")
print(feat_imp.to_string(index=False))

# COMPUTE RESIDUALS FOR ALL PAPERS
# ALGORITHM:
# 1. Predict log citations for ALL 36,823 papers (not just test set)
# 2. Convert predicted log citations back to citation scale:
#    predicted_citations = exp(predicted_log) - 1
#    (inverse of log1p)
# 3. Compute residual in log space (more stable):
#    residual = actual_log_citations - predicted_log_citations
#    Negative residual = got fewer citations than predicted = under-cited
#    Positive residual = got more citations than predicted = over-cited
#
# WHY LOG SPACE RESIDUALS?
#   In original citation space, a paper predicted to get 100 citations
#   but getting 50 has residual -50.
#   A paper predicted to get 5 but getting 0 has residual -5.
#   The first paper seems much more under-cited but proportionally
#   they're equally under-cited (both got 50% of prediction).
#   In log space: log(51) - log(101) ≈ log(1) - log(6) ≈ similar.
#   Log space treats proportional under-citation equally regardless
#   of absolute citation count.
 
print("\n" + "=" * 60)
print("Computing Residuals for All Papers")
print("=" * 60)
 
# Predict on ALL papers
y_pred_all = model.predict(X)
 
df["predicted_log_citations"] = y_pred_all
df["predicted_citations"]     = np.expm1(y_pred_all)   # inverse of log1p
df["residual"]                = df["log_citations"] - df["predicted_log_citations"]
 
# Positive residual = over-cited (got more than expected)
# Negative residual = under-cited (got less than expected)
 
print(f"Residual distribution (all papers):")
print(f"  Mean residual:      {df['residual'].mean():.4f}  (should be near 0)")
print(f"  Std residual:       {df['residual'].std():.4f}")
print(f"  Min residual:       {df['residual'].min():.4f}  (most under-cited)")
print(f"  Max residual:       {df['residual'].max():.4f}  (most over-cited)")
print(f"  Under-cited papers: {(df['residual'] < 0).sum():,}")
print(f"  Over-cited papers:  {(df['residual'] > 0).sum():,}")
 
# Residuals for bridge vs non-bridge
bridge_residuals    = df[df["is_bridge"]]["residual"]
nonbridge_residuals = df[~df["is_bridge"]]["residual"]
 
print(f"\nResiduals — Bridge papers (n={len(bridge_residuals):,}):")
print(f"  Mean:    {bridge_residuals.mean():.4f}")
print(f"  Median:  {bridge_residuals.median():.4f}")
 
print(f"\nResiduals — Non-bridge papers (n={len(nonbridge_residuals):,}):")
print(f"  Mean:    {nonbridge_residuals.mean():.4f}")
print(f"  Median:  {nonbridge_residuals.median():.4f}")
 
# This is the key preview of the Wilcoxon test result:
diff = bridge_residuals.mean() - nonbridge_residuals.mean()
print(f"\nMean residual difference (bridge - non-bridge): {diff:.4f}")
if diff < 0:
    print("  → Bridge papers have MORE NEGATIVE residuals.")
    print("  → Preliminary evidence: bridge papers are under-cited.")
else:
    print("  → Bridge papers have less negative residuals than non-bridge.")
    print("  → Check pipeline — this is unexpected.")


# FEATURE IMPORTANCE PLOT

print("\nSaving feature importance plot...")
 
fig, ax = plt.subplots(figsize=(8, 5))
feat_imp_sorted = feat_imp.sort_values("importance")
ax.barh(feat_imp_sorted["feature"], feat_imp_sorted["importance"],
        color="steelblue")
ax.set_xlabel("Feature Importance (LightGBM splits)")
ax.set_title("What predicts citation count?")
ax.axvline(x=0, color="black", linewidth=0.5)
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {plot_path}")

# SAVE OUTPUT
out_path = os.path.join(OUTPUT_DIR, "papers_with_residuals.parquet")
df.to_parquet(out_path, index=False)
print(f"\nSaved → {out_path}")
 
stats = f"""
=== Citation Regression Stats ===
 
MODEL
  Algorithm:          LightGBM Gradient Boosting Regressor
  Target:             log(1 + num_citations)
  Features:           {feature_cols}
  Train size:         {len(X_train):,}
  Test size:          {len(X_test):,}
 
PERFORMANCE
  R² score:           {r2:.4f}
  RMSE (log scale):   {rmse:.4f}
  Best iteration:     {model.best_iteration_}
 
FEATURE IMPORTANCES:
{feat_imp.to_string(index=False)}
 
RESIDUALS
  Mean (all):         {df['residual'].mean():.4f}
  Std (all):          {df['residual'].std():.4f}
  Under-cited papers: {(df['residual'] < 0).sum():,}
  Over-cited papers:  {(df['residual'] > 0).sum():,}
 
BRIDGE vs NON-BRIDGE RESIDUALS
  Bridge papers (n={len(bridge_residuals):,}):
    Mean residual:    {bridge_residuals.mean():.4f}
    Median residual:  {bridge_residuals.median():.4f}
  Non-bridge papers (n={len(nonbridge_residuals):,}):
    Mean residual:    {nonbridge_residuals.mean():.4f}
    Median residual:  {nonbridge_residuals.median():.4f}
  Difference:         {diff:.4f}
"""
 
stats_path = os.path.join(OUTPUT_DIR, "regression_stats.txt")
with open(stats_path, "w") as f:
    f.write(stats)
print(f"Stats saved → {stats_path}")
 
print("\nDone. Stage 3 complete.")
print("\nNew columns in papers_with_residuals.parquet:")
print("  age                     — years since publication")
print("  community_size          — papers in same community")
print("  log_citations           — log(1 + num_citations)")
print("  predicted_log_citations — model prediction")
print("  predicted_citations     — predicted count (original scale)")
print("  residual                — actual - predicted (log scale)")
print("                            negative = under-cited")
print("\nHand papers_with_residuals.parquet to Stage 5 (Wilcoxon test).")
print("Also usable by Stage 4 (SciBERT) for abstract validation.")