import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

# load GPT ratings file (has human ratings already)
gpt_df = pd.read_csv("results_gpt_judge.csv")

# load metrics file to get clipscore
metrics_df = pd.read_csv("results_human_v2.csv")

# merge on description + context + img_id to get clipscore
merged = gpt_df.merge(
    metrics_df[["img_id", "context", "description", "clipscore"]],
    on=["img_id", "context", "description"],
    how="left"
)
print(f"Merged: {len(merged)} rows")
print(f"Missing clipscore: {merged['clipscore'].isna().sum()}")

# drop rows missing key variables
analysis_df = merged.dropna(subset=[
    "clipscore", "gpt_relevance",
    "sighted_overall", "sighted_relevance",
    "blv_overall", "blv_relevance"
])
print(f"Rows with full data: {len(analysis_df)}")

# ── regression results ───────────────────────────────────────────────
for outcome, label in [
    ("sighted_overall",   "Sighted Overall"),
    ("sighted_relevance", "Sighted Relevance"),
    ("blv_overall",       "BLV Overall"),
    ("blv_relevance",     "BLV Relevance"),
]:
    df = analysis_df.dropna(subset=[outcome])
    X_full = sm.add_constant(df[["gpt_relevance", "clipscore"]])
    X_base = sm.add_constant(df[["clipscore"]])
    y = df[outcome]

    model_full = sm.OLS(y, X_full).fit()
    model_base = sm.OLS(y, X_base).fit()

    print(f"\n{label} (n={len(df)})")
    print(f"  GPT β={model_full.params['gpt_relevance']:.3f}, p={model_full.pvalues['gpt_relevance']:.3f}")
    print(f"  CLIPScore β={model_full.params['clipscore']:.3f}, p={model_full.pvalues['clipscore']:.3f}")
    print(f"  R² full={model_full.rsquared:.3f}, base={model_base.rsquared:.3f}, ΔR²={model_full.rsquared - model_base.rsquared:.3f}")

# ── partial regression plots ─────────────────────────────────────────
ratings  = ["sighted_overall", "sighted_relevance", "blv_overall", "blv_relevance"]
ylabels  = ["Sighted Overall", "Sighted Relevance", "BLV Overall", "BLV Relevance"]
colors   = ["#4C72B0", "#4C72B0", "#DD8452", "#DD8452"]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(
    "GPT-4o Relevance vs. Human Ratings\n(Controlling for CLIPScore via Partial Regression)",
    fontsize=13, y=1.02)

for ax, (rating, ylabel, color) in zip(axes, zip(ratings, ylabels, colors)):
    df = analysis_df.dropna(subset=[rating])
    X_ctrl = sm.add_constant(df[["clipscore"]])

    gpt_resid     = sm.OLS(df["gpt_relevance"], X_ctrl).fit().resid
    outcome_resid = sm.OLS(df[rating],          X_ctrl).fit().resid

    r, p = stats.pearsonr(gpt_resid, outcome_resid)

    # jitter since GPT scores are integers
    x_jitter = gpt_resid + np.random.uniform(-0.1, 0.1, size=len(gpt_resid))
    ax.scatter(x_jitter, outcome_resid, alpha=0.5, s=35,
               color=color, edgecolors="white", linewidth=0.4)

    m, b   = np.polyfit(gpt_resid, outcome_resid, 1)
    x_line = np.array([gpt_resid.min(), gpt_resid.max()])
    ax.plot(x_line, m*x_line+b, color="tomato", linewidth=2)

    ax.set_xlabel("GPT-4o Relevance\n(residualized on CLIPScore)", fontsize=10)
    ax.set_ylabel(f"{ylabel}\n(residualized on CLIPScore)", fontsize=10)
    ax.set_title(f"Partial r={r:.2f} (p={p:.3f})\nn={len(df)}", fontsize=10)
    ax.grid(True, alpha=0.3)

fig.text(0.5, -0.04,
         "Blue = Sighted participants    Orange = BLV participants",
         ha="center", fontsize=10, color="gray")

plt.tight_layout()
plt.savefig("gpt_partial_regression.png", dpi=150, bbox_inches="tight")
print("\nSaved gpt_partial_regression.png")