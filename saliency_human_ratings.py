import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# load saliency scores (one per image-context pair, use image-level by taking first per img_file)
sal_df = pd.read_csv("results_three_axes_v2.csv")
# get one saliency value per unique image (same across contexts)
sal_per_image = sal_df[["img_file", "saliency"]].drop_duplicates(subset="img_file")
# apply threshold
sal_per_image = sal_per_image.copy()
sal_per_image.loc[sal_per_image["saliency"] < 0.05, "saliency"] = None

# load human descriptions with ratings
human_df = pd.read_csv("results_human_v2.csv")

# load sighted data to get reconstructivity preimg
sighted_df = pd.read_csv("behavioral_data/sighted_data_criticaltrials.csv")
sighted_agg = sighted_df.groupby(
    ["img_id", "context", "description"]
).agg(
    sighted_reconstruct_pre  = ("q_reconstructivity.preimg", "mean"),
    sighted_overall_post     = ("q_overall.postimg",         "mean"),
).reset_index()

# merge everything
merged = human_df.merge(sal_per_image, on="img_file", how="left")
merged = merged.merge(sighted_agg, on=["img_id", "context", "description"], how="left")

plot_df = merged.dropna(subset=[
    "saliency",
    "sighted_reconstruct_pre",
    "sighted_overall_post",
    "sighted_reconstruct",  # this is preimg from results_human_v2
])
print(f"Plotting {len(plot_df)} rows")

# three panels
outcomes = [
    ("sighted_reconstruct_pre", "Sighted Reconstructivity (pre-image)"),
    ("sighted_reconstruct",     "Sighted Reconstructivity (pre-image, v2)"),
    ("sighted_overall_post",    "Sighted Overall (post-image)"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Saliency Alignment vs. Sighted Human Ratings\n(Human-Written Descriptions)", fontsize=13)

for ax, (outcome, label) in zip(axes, outcomes):
    df = plot_df.dropna(subset=[outcome])
    x  = df["saliency"]
    y  = df[outcome]

    pearson_r,  pearson_p  = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    ax.scatter(x, y, alpha=0.5, s=35, color="steelblue",
               edgecolors="white", linewidth=0.4)
    m, b   = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    ax.plot(x_line, m*x_line+b, color="tomato", linewidth=2)

    ax.set_xlabel("Saliency Alignment", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(
        f"r={pearson_r:.2f} (p={pearson_p:.3f})\n"
        f"ρ={spearman_r:.2f} (p={spearman_p:.3f})\n"
        f"n={len(df)}",
        fontsize=10
    )
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("saliency_sighted_ratings.png", dpi=150, bbox_inches="tight")
print("Saved saliency_sighted_ratings.png")