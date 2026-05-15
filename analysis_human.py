import os, torch, clip, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from scipy import stats

IMAGES_DIR = "metrics/clipscore/cosid_data/cosid_images"
DESC_CSV   = "behavioral_data/all_descriptions.csv"
SIGHTED_CSV = "behavioral_data/sighted_data_criticaltrials.csv"
BLV_CSV    = "behavioral_data/blv_data_criticaltrials.csv"
OUTPUT_CSV = "results_human.csv"

clip_model = preprocess = sbert = None

def setup():
    global clip_model, preprocess, sbert
    clip_model, preprocess = clip.load("ViT-B/32", device="cpu")
    sbert = SentenceTransformer("all-mpnet-base-v2")

def clipscore(image_path, caption):
    img = preprocess(Image.open(image_path)).unsqueeze(0)
    txt = clip.tokenize([caption], truncate=True)
    if isinstance(txt, dict):
        txt = list(txt.values())[0]
    with torch.no_grad():
        img_feat = clip_model.encode_image(img)
        txt_feat = clip_model.encode_text(txt)
    img_feat /= img_feat.norm(dim=-1, keepdim=True)
    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
    return 2.5 * max(0, (img_feat @ txt_feat.T).item())

def context_sim(context_text, caption):
    e1, e2 = sbert.encode([context_text, caption])
    return util.cos_sim(e1, e2).item()

def get_image_path(img_file):
    img_stem = os.path.splitext(img_file)[0]
    candidates = [f for f in os.listdir(IMAGES_DIR) if f.startswith(img_stem)]
    if candidates:
        return os.path.join(IMAGES_DIR, candidates[0])
    return None

def main():
    print("Setting up models...")
    setup()

    # load data
    desc_df    = pd.read_csv(DESC_CSV)
    sighted_df = pd.read_csv(SIGHTED_CSV)
    blv_df     = pd.read_csv(BLV_CSV)

    # aggregate human ratings per (img_id, context, description)
    sighted_agg = sighted_df.groupby(
        ["img_id", "context", "description"]
    ).agg(
        sighted_overall    = ("q_overall.postimg",       "mean"),
        sighted_relevance  = ("q_relevance.postimg",     "mean"),
        sighted_reconstruct= ("q_reconstructivity.preimg","mean"),
    ).reset_index()

    blv_agg = blv_df.groupby(
        ["img_id", "context", "description"]
    ).agg(
        blv_overall    = ("q_overall",        "mean"),
        blv_relevance  = ("q_relevance",      "mean"),
        blv_reconstruct= ("q_reconstructivity","mean"),
    ).reset_index()

    # merge ratings onto descriptions
    merged = desc_df.merge(sighted_agg, on=["img_id", "context", "description"], how="left")
    merged = merged.merge(blv_agg,     on=["img_id", "context", "description"], how="left")
    print(f"Merged dataset: {len(merged)} rows")

    # compute CLIPScore and context_sim per description
    clipscores   = []
    context_sims = []

    for i, row in merged.iterrows():
        img_path = get_image_path(row["img_file"])
        if img_path is None:
            print(f"MISSING: {row['img_file']}")
            clipscores.append(None)
            context_sims.append(None)
            continue
        if i % 50 == 0:
            print(f"  Processing row {i}/{len(merged)}...")
        clipscores.append(clipscore(img_path, row["description"]))
        context_sims.append(context_sim(row["article_text"], row["description"]))

    merged["clipscore"]   = clipscores
    merged["context_sim"] = context_sims
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")

    # drop rows missing ratings or scores
    plot_df = merged.dropna(subset=[
        "clipscore", "context_sim",
        "sighted_overall", "blv_overall"
    ])
    print(f"Rows with full data for plotting: {len(plot_df)}")

    # plot
    metrics  = ["clipscore", "context_sim"]
    ratings  = ["sighted_overall", "sighted_relevance", "blv_overall", "blv_relevance"]
    ylabels  = ["Sighted Overall", "Sighted Relevance", "BLV Overall", "BLV Relevance"]
    xlabels  = ["CLIPScore", "Context Similarity"]

    fig, axes = plt.subplots(len(ratings), len(metrics), figsize=(12, 16))
    fig.suptitle("Human Ratings vs. Automated Metrics (Human-Written Descriptions)", fontsize=14, y=1.01)

    for row_i, (rating, ylabel) in enumerate(zip(ratings, ylabels)):
        for col_i, (metric, xlabel) in enumerate(zip(metrics, xlabels)):
            ax = axes[row_i][col_i]
            x = plot_df[metric]
            y = plot_df[rating]

            # pearson and spearman
            pearson_r,  pearson_p  = stats.pearsonr(x, y)
            spearman_r, spearman_p = stats.spearmanr(x, y)

            ax.scatter(x, y, alpha=0.4, s=30, color="steelblue", edgecolors="white", linewidth=0.3)

            # regression line
            m, b = np.polyfit(x, y, 1)
            x_line = pd.Series([x.min(), x.max()])
            ax.plot(x_line, m * x_line + b, color="tomato", linewidth=1.5)

            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(
                f"r={pearson_r:.2f} (p={pearson_p:.3f})\nρ={spearman_r:.2f} (p={spearman_p:.3f})",
                fontsize=9
            )
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("human_ratings_vs_metrics.png", dpi=150, bbox_inches="tight")
    print("Saved plot to human_ratings_vs_metrics.png")

if __name__ == "__main__":
    import numpy as np
    main()