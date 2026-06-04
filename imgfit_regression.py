import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

DESC_CSV    = "behavioral_data/all_descriptions.csv"
SIGHTED_CSV = "behavioral_data/sighted_data_criticaltrials.csv"
BLV_CSV     = "behavioral_data/blv_data_criticaltrials.csv"
HUMAN_V2    = "results_human_v2.csv"  # has clipscore and bertscore_f already

def main():
    # load existing metric scores for human descriptions
    metrics_df = pd.read_csv(HUMAN_V2)
    print(f"Loaded metrics: {len(metrics_df)} rows")
    print(f"Columns: {list(metrics_df.columns)}")

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

    # merge onto metrics
    merged = metrics_df.merge(sighted_imgfit,
                              on=["img_id", "context", "description"],
                              how="left")
    merged = merged.merge(blv_imgfit,
                          on=["img_id", "context", "description"],
                          how="left")
    print(f"After merge: {len(merged)} rows")

    # drop rows missing any key variable
    analysis_df = merged.dropna(subset=[
        "clipscore", "bertscore_f",
        "sighted_imgfit_post", "blv_imgfit"
    ])
    print(f"Rows with full data: {len(analysis_df)}")

    # ── run regressions ──────────────────────────────────────────────────
    results = {}
    for outcome, label in [
        ("sighted_imgfit_post", "Sighted Imgfit (post)"),
        ("sighted_imgfit_pre",  "Sighted Imgfit (pre)"),
        ("blv_imgfit",          "BLV Imgfit"),
    ]:
        df = analysis_df.dropna(subset=[outcome])
        X = sm.add_constant(df[["bertscore_f", "clipscore"]])
        y = df[outcome]

        # full model
        model_full     = sm.OLS(y, X).fit()
        # baseline (clipscore only)
        X_base         = sm.add_constant(df[["clipscore"]])
        model_base     = sm.OLS(y, X_base).fit()

        results[outcome] = {
            "label":       label,
            "n":           len(df),
            "beta_bert":   model_full.params["bertscore_f"],
            "p_bert":      model_full.pvalues["bertscore_f"],
            "beta_clip":   model_full.params["clipscore"],
            "p_clip":      model_full.pvalues["clipscore"],
            "r2_full":     model_full.rsquared,
            "r2_base":     model_base.rsquared,
            "r2_delta":    model_full.rsquared - model_base.rsquared,
        }

        print(f"\n{'='*50}")
        print(f"{label} (n={len(df)})")
        print(f"  BERTScore β={results[outcome]['beta_bert']:.3f}, p={results[outcome]['p_bert']:.3f}")
        print(f"  CLIPScore β={results[outcome]['beta_clip']:.3f}, p={results[outcome]['p_clip']:.3f}")
        print(f"  R² full={results[outcome]['r2_full']:.3f}, baseline={results[outcome]['r2_base']:.3f}, ΔR²={results[outcome]['r2_delta']:.3f}")

    # ── plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "BERTScore Recall vs Image-Article Fit\n(Controlling for CLIPScore via Partial Regression)",
        fontsize=13)

    for ax, (outcome, label) in zip(axes, [
        ("sighted_imgfit_post", "Sighted Imgfit (post-image)"),
        ("sighted_imgfit_pre",  "Sighted Imgfit (pre-image)"),
        ("blv_imgfit",          "BLV Imgfit"),
    ]):
        df = analysis_df.dropna(subset=[outcome])

        # partial regression plot — residuals of BERTScore ~ CLIPScore
        # vs residuals of outcome ~ CLIPScore
        X_ctrl = sm.add_constant(df[["clipscore"]])

        bert_resid    = sm.OLS(df["bertscore_f"], X_ctrl).fit().resid
        outcome_resid = sm.OLS(df[outcome],       X_ctrl).fit().resid

        r, p = stats.pearsonr(bert_resid, outcome_resid)

        ax.scatter(bert_resid, outcome_resid, alpha=0.5, s=35,
                   color="steelblue", edgecolors="white", linewidth=0.3)
        m, b   = np.polyfit(bert_resid, outcome_resid, 1)
        x_line = np.array([bert_resid.min(), bert_resid.max()])
        ax.plot(x_line, m*x_line+b, color="tomato", linewidth=2)

        ax.set_xlabel("BERTScore Recall\n(residualized on CLIPScore)", fontsize=10)
        ax.set_ylabel(f"{label}\n(residualized on CLIPScore)", fontsize=10)
        ax.set_title(
            f"Partial r={r:.2f} (p={p:.3f})\nn={len(df)}",
            fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("imgfit_partial_regression.png", dpi=150, bbox_inches="tight")
    print("\nSaved imgfit_partial_regression.png")

if __name__ == "__main__":
    main()