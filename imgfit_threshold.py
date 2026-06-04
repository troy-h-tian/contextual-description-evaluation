import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

DESC_CSV    = "behavioral_data/all_descriptions.csv"
SIGHTED_CSV = "behavioral_data/sighted_data_criticaltrials.csv"
BLV_CSV     = "behavioral_data/blv_data_criticaltrials.csv"
HUMAN_V2    = "results_human_v2.csv"

def main():
    # load metrics
    metrics_df = pd.read_csv(HUMAN_V2)
    print(f"Loaded metrics: {len(metrics_df)} rows")

    # load and aggregate imgfit ratings
    sighted_df = pd.read_csv(SIGHTED_CSV)
    blv_df     = pd.read_csv(BLV_CSV)

    sighted_imgfit = sighted_df.groupby(
        ["img_id", "context", "description"]
    ).agg(
        sighted_imgfit_pre  = ("q_imgfit.preimg",  "mean"),
        sighted_imgfit_post = ("q_imgfit.postimg", "mean"),
    ).reset_index()

    blv_imgfit = blv_df.groupby(
        ["img_id", "context", "description"]
    ).agg(
        blv_imgfit = ("q_imgfit", "mean"),
    ).reset_index()

    merged = metrics_df.merge(sighted_imgfit,
                              on=["img_id", "context", "description"],
                              how="left")
    merged = merged.merge(blv_imgfit,
                          on=["img_id", "context", "description"],
                          how="left")

    full_df = merged.dropna(subset=["clipscore", "bertscore_f",
                                     "sighted_imgfit_post", "blv_imgfit"])
    print(f"Rows with full data: {len(full_df)}")

    # compute 75th percentile CLIPScore threshold
    threshold = full_df["clipscore"].quantile(0.75)
    high_clip_df = full_df[full_df["clipscore"] >= threshold]
    print(f"CLIPScore 75th percentile threshold: {threshold:.3f}")
    print(f"Rows above threshold: {len(high_clip_df)}")

    # ── correlations on restricted sample ───────────────────────────────
    outcomes = [
        ("sighted_imgfit_post", "Sighted Imgfit\n(post-image)"),
        ("sighted_imgfit_pre",  "Sighted Imgfit\n(pre-image)"),
        ("blv_imgfit",          "BLV Imgfit"),
    ]

    print(f"\nCorrelations on high-CLIPScore subset (n={len(high_clip_df)}):")
    for outcome, label in outcomes:
        df = high_clip_df.dropna(subset=[outcome])
        r, p = stats.pearsonr(df["bertscore_f"], df[outcome])
        rho, p_rho = stats.spearmanr(df["bertscore_f"], df[outcome])
        print(f"  {label.replace(chr(10), ' ')}: r={r:.2f} (p={p:.3f}), ρ={rho:.2f} (p={p_rho:.3f}), n={len(df)}")

    # also print full sample for comparison
    print(f"\nCorrelations on full sample (n={len(full_df)}) for comparison:")
    for outcome, label in outcomes:
        df = full_df.dropna(subset=[outcome])
        r, p = stats.pearsonr(df["bertscore_f"], df[outcome])
        rho, p_rho = stats.spearmanr(df["bertscore_f"], df[outcome])
        print(f"  {label.replace(chr(10), ' ')}: r={r:.2f} (p={p:.3f}), ρ={rho:.2f} (p={p_rho:.3f}), n={len(df)}")

    # ── plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"BERTScore Recall vs Image-Article Fit\n"
        f"(High-CLIPScore Descriptions Only, threshold={threshold:.2f}, n={len(high_clip_df)})",
        fontsize=13)

    for ax, (outcome, label) in zip(axes, outcomes):
        df = high_clip_df.dropna(subset=[outcome])
        x  = df["bertscore_f"]
        y  = df[outcome]

        r, p     = stats.pearsonr(x, y)
        rho, p_rho = stats.spearmanr(x, y)

        ax.scatter(x, y, alpha=0.5, s=35, color="steelblue",
                   edgecolors="white", linewidth=0.3)
        m, b   = np.polyfit(x, y, 1)
        x_line = np.array([x.min(), x.max()])
        ax.plot(x_line, m*x_line+b, color="tomato", linewidth=2)

        ax.set_xlabel("BERTScore Recall", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(
            f"r={r:.2f} (p={p:.3f})\nρ={rho:.2f} (p={p_rho:.3f}), n={len(df)}",
            fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("imgfit_threshold.png", dpi=150, bbox_inches="tight")
    print("\nSaved imgfit_threshold.png")

if __name__ == "__main__":
    main()