import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

# load GPT ratings (has human ratings already)
gpt_df = pd.read_csv("results_gpt_judge.csv")

# load metrics to get clipscore
metrics_df = pd.read_csv("results_human_v2.csv")

# merge to get clipscore
merged = gpt_df.merge(
    metrics_df[["img_id", "context", "description", "clipscore"]],
    on=["img_id", "context", "description"],
    how="left"
)

# load imgfit ratings
sighted_df = pd.read_csv("behavioral_data/sighted_data_criticaltrials.csv")
blv_df     = pd.read_csv("behavioral_data/blv_data_criticaltrials.csv")

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

merged = merged.merge(sighted_imgfit, on=["img_id", "context", "description"], how="left")
merged = merged.merge(blv_imgfit,     on=["img_id", "context", "description"], how="left")

analysis_df = merged.dropna(subset=[
    "clipscore", "gpt_relevance",
    "sighted_imgfit_post", "blv_imgfit"
])
print(f"Rows with full data: {len(analysis_df)}")

# ── partial regression plots ─────────────────────────────────────────
outcomes = [
    ("sighted_imgfit_post", "Sighted Imgfit (post-image)"),
    ("sighted_imgfit_pre",  "Sighted Imgfit (pre-image)"),
    ("blv_imgfit",          "BLV Imgfit"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "GPT-4o Relevance vs. Image-Article Fit\n(Controlling for CLIPScore via Partial Regression)",
    fontsize=13)

for ax, (outcome, label) in zip(axes, outcomes):
    df = analysis_df.dropna(subset=[outcome])
    X_ctrl = sm.add_constant(df[["clipscore"]])

    gpt_resid     = sm.OLS(df["gpt_relevance"], X_ctrl).fit().resid
    outcome_resid = sm.OLS(df[outcome],         X_ctrl).fit().resid

    r, p = stats.pearsonr(gpt_resid, outcome_resid)

    x_jitter = gpt_resid + np.random.uniform(-0.1, 0.1, size=len(gpt_resid))
    ax.scatter(x_jitter, outcome_resid, alpha=0.5, s=35,
               color="steelblue", edgecolors="white", linewidth=0.4)

    m, b   = np.polyfit(gpt_resid, outcome_resid, 1)
    x_line = np.array([gpt_resid.min(), gpt_resid.max()])
    ax.plot(x_line, m*x_line+b, color="tomato", linewidth=2)

    ax.set_xlabel("GPT-4o Relevance\n(residualized on CLIPScore)", fontsize=10)
    ax.set_ylabel(f"{label}\n(residualized on CLIPScore)", fontsize=10)
    ax.set_title(f"Partial r={r:.2f} (p={p:.3f})\nn={len(df)}", fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gpt_imgfit_partial.png", dpi=150, bbox_inches="tight")
print("Saved gpt_imgfit_partial.png")