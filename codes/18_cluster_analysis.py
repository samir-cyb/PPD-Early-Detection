"""
18_cluster_analysis.py
=========================
Improvement idea: UNSUPERVISED CLUSTER ANALYSIS, done BEFORE/separately from
the supervised prediction task.

Every prior script asked "given a mother's profile, predict her PPD risk."
This script asks a different question: "ignoring PPD entirely, do mothers
naturally fall into distinct psychosocial risk PROFILES / groups?" -- and
only AFTER discovering those groups, checks whether PPD prevalence differs
across them. This can surface a pattern that supervised classification
doesn't directly show: e.g. "there's a natural cluster of young mothers
with low economic stability and low family support, and PPD is dramatically
more common in that cluster than any other" -- a finding a clinician/policy
maker can act on directly (target that specific profile for outreach),
which is a different and complementary kind of insight to per-feature SHAP
importance.

Clustering is done on the 4 WEIGHTED composite indices + Age + Number of
the latest pregnancy (6 continuous, clinically interpretable dimensions) --
NOT on the full 112-dimension one-hot feature space, because (as the SMOTE
experiment in script 14 already demonstrated) distance-based methods behave
poorly in high-dimensional mostly-binary spaces. The composite indices are
exactly the right compact summary for this: they already distill the raw
answers into 4 clinically meaningful continuous scores.

Output (./outputs/):
  - cluster_silhouette_scores.csv     -- which K was tried, which chosen
  - cluster_profiles.csv              -- mean composite/Age/pregnancy-count
                                          per cluster
  - cluster_ppd_rates.csv             -- PPD_binary rate, EPDS/PHQ9
                                          distribution per cluster + chi2 test
  - cluster_pca_scatter.png           -- 2D visualization, colored by
                                          cluster and by PPD_binary
  - cluster_analysis_summary.txt

Usage
-----
    python 18_cluster_analysis.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "outputs")
MODEL_READY_PATH = os.path.join(OUT_DIR, "PPD_model_ready.csv")

if not os.path.exists(MODEL_READY_PATH):
    raise SystemExit(f"{MODEL_READY_PATH} not found. Run 06_encoding.py first.")

df = pd.read_csv(MODEL_READY_PATH)
print(f"Loaded model-ready dataset: {df.shape}")

RANDOM_STATE = 42
CLUSTER_FEATURES = [
    "Economic_Stability_Index_Weighted", "Social_Support_Index_Weighted",
    "Maternal_MentalHealth_Risk_Index_Weighted",
    "Neonatal_Delivery_Stress_Index_Weighted",
    "Age", "Number of the latest pregnancy",
]
print(f"Clustering on: {CLUSTER_FEATURES}")

X = df[CLUSTER_FEATURES].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------------------
# 1. Choose K via silhouette score (try K=2..6)
# ---------------------------------------------------------------------------
print(f"\n{'=' * 90}\nCHOOSING K VIA SILHOUETTE SCORE\n{'=' * 90}")
sil_rows = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_rows.append({"k": k, "silhouette_score": score})
    print(f"  K={k}: silhouette = {score:.4f}")

sil_df = pd.DataFrame(sil_rows)
best_k = int(sil_df.loc[sil_df["silhouette_score"].idxmax(), "k"])
print(f"\nBest K by silhouette score: {best_k}")
sil_path = os.path.join(OUT_DIR, "cluster_silhouette_scores.csv")
sil_df.to_csv(sil_path, index=False)
print(f"Saved -> {sil_path}")

# ---------------------------------------------------------------------------
# 2. Fit final KMeans, profile clusters
# ---------------------------------------------------------------------------
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print(f"\n{'=' * 90}\nCLUSTER PROFILES (K={best_k})\n{'=' * 90}")
profile_df = df.groupby("Cluster")[CLUSTER_FEATURES].mean().round(2)
profile_df["n_mothers"] = df.groupby("Cluster").size()
print(profile_df.to_string())
profile_path = os.path.join(OUT_DIR, "cluster_profiles.csv")
profile_df.to_csv(profile_path)
print(f"Saved -> {profile_path}")

# Auto-generate a descriptive label per cluster from its own profile,
# relative to the overall mean (no hard-coded assumptions about which
# cluster is "the risky one" -- purely data-driven).
overall_mean = df[CLUSTER_FEATURES].mean()
cluster_labels = {}
for c in profile_df.index:
    tags = []
    if profile_df.loc[c, "Social_Support_Index_Weighted"] < overall_mean["Social_Support_Index_Weighted"]:
        tags.append("Lower support")
    else:
        tags.append("Higher support")
    if profile_df.loc[c, "Maternal_MentalHealth_Risk_Index_Weighted"] > overall_mean["Maternal_MentalHealth_Risk_Index_Weighted"]:
        tags.append("higher mental-health risk")
    else:
        tags.append("lower mental-health risk")
    if profile_df.loc[c, "Economic_Stability_Index_Weighted"] < overall_mean["Economic_Stability_Index_Weighted"]:
        tags.append("lower economic stability")
    else:
        tags.append("higher economic stability")
    cluster_labels[c] = ", ".join(tags)
    print(f"  Cluster {c} ({int(profile_df.loc[c, 'n_mothers'])} mothers): {tags}")

# ---------------------------------------------------------------------------
# 3. PPD rate per cluster + significance test
# ---------------------------------------------------------------------------
print(f"\n{'=' * 90}\nPPD PREVALENCE PER CLUSTER\n{'=' * 90}")
ppd_rate = df.groupby("Cluster")["PPD_binary"].mean().round(3)
print("PPD_binary positive rate per cluster:")
print(ppd_rate.to_string())

contingency = pd.crosstab(df["Cluster"], df["PPD_binary"])
chi2, p_value, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square test (Cluster vs PPD_binary): chi2={chi2:.2f}, "
      f"p={p_value:.2e}, {'SIGNIFICANT' if p_value < 0.05 else 'not significant'} "
      f"at p<0.05")

epds_dist = pd.crosstab(df["Cluster"], df["EPDS Result"], normalize="index").round(3)
phq9_dist = pd.crosstab(df["Cluster"], df["PHQ9 Result"], normalize="index").round(3)
print("\nEPDS Result distribution per cluster (row %):")
print(epds_dist.to_string())
print("\nPHQ9 Result distribution per cluster (row %):")
print(phq9_dist.to_string())

rates_df = ppd_rate.to_frame("PPD_binary_rate")
rates_df["cluster_label"] = pd.Series(cluster_labels)
rates_df["n_mothers"] = profile_df["n_mothers"]
rates_path = os.path.join(OUT_DIR, "cluster_ppd_rates.csv")
rates_df.to_csv(rates_path)
print(f"\nSaved -> {rates_path}")

# ---------------------------------------------------------------------------
# 4. 2D visualization (PCA)
# ---------------------------------------------------------------------------
try:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scatter1 = axes[0].scatter(coords[:, 0], coords[:, 1], c=df["Cluster"],
                                cmap="tab10", alpha=0.6, s=20)
    axes[0].set_title(f"Colored by Cluster (K={best_k})")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.colorbar(scatter1, ax=axes[0], label="Cluster")

    scatter2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=df["PPD_binary"],
                                cmap="coolwarm", alpha=0.6, s=20)
    axes[1].set_title("Colored by PPD_binary (same 2D space)")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.colorbar(scatter2, ax=axes[1], label="PPD_binary")

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "cluster_pca_scatter.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved -> {plot_path}")
except Exception as e:
    print(f"[plot failed] PCA scatter: {e}")

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
highest_risk_cluster = ppd_rate.idxmax()
lowest_risk_cluster = ppd_rate.idxmin()
summary = (
    f"Cluster analysis summary (K={best_k}, chosen by silhouette score)\n"
    f"{'=' * 60}\n"
    f"Cluster {highest_risk_cluster} has the HIGHEST PPD rate "
    f"({ppd_rate[highest_risk_cluster]:.1%}): {cluster_labels[highest_risk_cluster]}\n"
    f"Cluster {lowest_risk_cluster} has the LOWEST PPD rate "
    f"({ppd_rate[lowest_risk_cluster]:.1%}): {cluster_labels[lowest_risk_cluster]}\n"
    f"Chi-square test (cluster vs PPD_binary): p={p_value:.2e} "
    f"({'significant' if p_value < 0.05 else 'not significant'})\n"
)
summary_path = os.path.join(OUT_DIR, "cluster_analysis_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary)
print(f"\n{'=' * 90}\nSUMMARY\n{'=' * 90}")
print(summary)
print(f"Saved -> {summary_path}")
print("\nDone. If the chi-square test is significant and the PPD-rate spread "
      "across clusters is large, this is a strong, independent (unsupervised) "
      "confirmation that the composite-index risk profile genuinely "
      "separates high- and low-risk groups of mothers -- useful for a "
      "'which mothers to prioritize for outreach' discussion.")
