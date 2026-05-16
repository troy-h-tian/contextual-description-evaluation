import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from openai import OpenAI

DESC_CSV    = "behavioral_data/all_descriptions.csv"
SIGHTED_CSV = "behavioral_data/sighted_data_criticaltrials.csv"
BLV_CSV     = "behavioral_data/blv_data_criticaltrials.csv"
OUTPUT_CSV  = "results_gpt_judge.csv"
OPENAI_KEY  = "REDACTED"

def get_ratings():
    desc_df    = pd.read_csv(DESC_CSV)
    sighted_df = pd.read_csv(SIGHTED_CSV)
    blv_df     = pd.read_csv(BLV_CSV)

    sighted_agg = sighted_df.groupby(
        ["img_id", "context", "description"]
    ).agg(
        sighted_overall     = ("q_overall.postimg",        "mean"),
        sighted_relevance   = ("q_relevance.postimg",      "mean"),
        sighted_reconstruct = ("q_reconstructivity.preimg","mean"),
    ).reset_index()

    blv_agg = blv_df.groupby(
        ["img_id", "context", "description"]
    ).agg(
        blv_overall     = ("q_overall",         "mean"),
        blv_relevance   = ("q_relevance",       "mean"),
        blv_reconstruct = ("q_reconstructivity","mean"),
    ).reset_index()

    merged = desc_df.merge(sighted_agg, on=["img_id","context","description"], how="left")
    merged = merged.merge(blv_agg,      on=["img_id","context","description"], how="left")
    return merged

def gpt_relevance(client, article_text, description, context_label):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""You are evaluating image descriptions for accessibility.

An image appears in a Wikipedia article about: {context_label}

Article context: {article_text[:500]}

Image description: {description}

Rate how relevant this description is for a blind or low vision user reading this specific article, on a scale of 1-5 where:
1 = completely irrelevant to the article context
3 = somewhat relevant
5 = highly relevant and useful given the article context

Respond with only a single integer from 1 to 5."""
        }],
        max_tokens=5
    )
    try:
        return int(response.choices[0].message.content.strip())
    except:
        return None

def run_gpt_ratings(merged):
    client = OpenAI(api_key=OPENAI_KEY)
    gpt_scores = []
    for i, row in merged.iterrows():
        if i % 50 == 0:
            print(f"  Row {i}/{len(merged)}...")
        try:
            gpt_scores.append(gpt_relevance(
                client,
                row["article_text"],
                row["description"],
                row["context"]
            ))
        except Exception as e:
            print(f"  ERROR row {i}: {e}")
            gpt_scores.append(None)
    merged["gpt_relevance"] = gpt_scores
    return merged

def plot_gpt_ratings(df, output_file="gpt_judge_plot.png"):
    plot_df = df.dropna(subset=[
        "gpt_relevance",
        "sighted_overall","sighted_relevance",
        "blv_overall","blv_relevance"
    ])
    print(f"Plotting {len(plot_df)} rows with full data")

    ratings = ["sighted_overall","sighted_relevance","blv_overall","blv_relevance"]
    ylabels = ["Sighted Overall","Sighted Relevance","BLV Overall","BLV Relevance"]
    colors  = ["#4C72B0","#4C72B0","#DD8452","#DD8452"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("GPT-4o Contextual Relevance vs. Human Ratings", fontsize=14, y=1.02)

    for i, (rating, ylabel, color) in enumerate(zip(ratings, ylabels, colors)):
        ax = axes[i]
        x  = plot_df["gpt_relevance"]
        y  = plot_df[rating]

        pearson_r,  pearson_p  = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        # jitter x slightly since GPT scores are integers
        x_jitter = x + np.random.uniform(-0.15, 0.15, size=len(x))
        ax.scatter(x_jitter, y, alpha=0.5, s=35, color=color,
                   edgecolors="white", linewidth=0.4)

        m, b   = np.polyfit(x, y, 1)
        x_line = np.array([x.min(), x.max()])
        ax.plot(x_line, m * x_line + b, color="tomato", linewidth=2)

        ax.set_xlabel("GPT-4o Relevance Score (1-5)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_title(
            f"r={pearson_r:.2f} (p={pearson_p:.3f})\n"
            f"ρ={spearman_r:.2f} (p={spearman_p:.3f})",
            fontsize=10
        )
        ax.grid(True, alpha=0.3)

    # sighted vs BLV divider
    fig.text(0.5, -0.04,
             "Blue = Sighted participants    Orange = BLV participants",
             ha="center", fontsize=10, color="gray")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_file}")

def main():
    print("Loading data...")
    merged = get_ratings()
    print(f"Loaded {len(merged)} rows")

    print("Running GPT ratings...")
    merged = run_gpt_ratings(merged)
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")

    plot_gpt_ratings(merged)

if __name__ == "__main__":
    main()