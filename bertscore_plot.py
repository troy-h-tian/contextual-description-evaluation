import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

df = pd.read_csv("results_human_v2.csv")

plot_df = df.dropna(subset=[
    "bertscore_f",
    "sighted_overall", "sighted_relevance",
    "blv_overall", "blv_relevance"
])
print(f"Plotting {len(plot_df)} rows")

ratings  = ["sighted_overall", "sighted_relevance", "blv_overall", "blv_relevance"]
ylabels  = ["Sighted Overall", "Sighted Relevance", "BLV Overall", "BLV Relevance"]
colors   = ["#4C72B0", "#4C72B0", "#DD8452", "#DD8452"]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("BERTScore F1 vs. Human Ratings (Human-Written Descriptions)", fontsize=14, y=1.02)

for i, (rating, ylabel, color) in enumerate(zip(ratings, ylabels, colors)):
    ax = axes[i]
    x  = plot_df["bertscore_f"]
    y  = plot_df[rating]

    pearson_r,  pearson_p  = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    ax.scatter(x, y, alpha=0.5, s=35, color=color,
               edgecolors="white", linewidth=0.4)

    m, b   = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    ax.plot(x_line, m * x_line + b, color="tomato", linewidth=2)

    ax.set_xlabel("BERTScore F1", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(
        f"r={pearson_r:.2f} (p={pearson_p:.3f})\n"
        f"ρ={spearman_r:.2f} (p={spearman_p:.3f})",
        fontsize=10
    )
    ax.grid(True, alpha=0.3)

fig.text(0.5, -0.04,
         "Blue = Sighted participants    Orange = BLV participants",
         ha="center", fontsize=10, color="gray")

plt.tight_layout()
plt.savefig("bertscore_human_ratings.png", dpi=150, bbox_inches="tight")
print("Saved bertscore_human_ratings.png")