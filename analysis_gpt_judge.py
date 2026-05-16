import os, torch, clip, pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from scipy import stats
from transformers import pipeline
from bert_score import score as bert_score
from openai import OpenAI

IMAGES_DIR  = "metrics/clipscore/cosid_data/cosid_images"
DESC_CSV    = "behavioral_data/all_descriptions.csv"
SIGHTED_CSV = "behavioral_data/sighted_data_criticaltrials.csv"
BLV_CSV     = "behavioral_data/blv_data_criticaltrials.csv"
OUTPUT_CSV  = "results_human_v3.csv"
OPENAI_KEY  = "REDACTED"

clip_model = preprocess = sbert = nli = client = None

def setup():
    global clip_model, preprocess, sbert, nli, client
    client = OpenAI(api_key=OPENAI_KEY)
    clip_model, preprocess = clip.load("ViT-B/32", device="cpu")
    sbert = SentenceTransformer("all-mpnet-base-v2")
    nli   = pipeline("text-classification",
                     model="cross-encoder/nli-deberta-v3-base",
                     device="cpu")

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

def nli_entailment(context_text, caption):
    premise = context_text[:512]
    result  = nli({"text": premise, "text_pair": caption})
    if isinstance(result, dict):
        result = [result]
    scores = {r["label"]: r["score"] for r in result}
    return scores.get("ENTAILMENT", scores.get("entailment", 0))

def gpt_relevance(article_text, description, context_label):
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

def get_image_path(img_file):
    img_stem   = os.path.splitext(img_file)[0]
    candidates = [f for f in os.listdir(IMAGES_DIR) if f.startswith(img_stem)]
    return os.path.join(IMAGES_DIR, candidates[0]) if candidates else None

def main():
    print("Setting up models...")
    setup()

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
    print(f"Merged: {len(merged)} rows")

    # compute automated metrics
    clipscores   = []
    context_sims = []
    entailments  = []
    gpt_scores   = []
    descriptions  = list(merged["description"])
    article_texts = list(merged["article_text"])

    for i, row in merged.iterrows():
        img_path = get_image_path(row["img_file"])
        if i % 50 == 0:
            print(f"  Row {i}/{len(merged)}...")
        if img_path is None:
            clipscores.append(None)
            context_sims.append(None)
            entailments.append(None)
            gpt_scores.append(None)
            continue
        clipscores.append(clipscore(img_path, row["description"]))
        context_sims.append(context_sim(row["article_text"], row["description"]))
        entailments.append(nli_entailment(row["article_text"], row["description"]))
        try:
            gpt_scores.append(gpt_relevance(
                row["article_text"],
                row["description"],
                row["context"]
            ))
        except Exception as e:
            print(f"  GPT ERROR row {i}: {e}")
            gpt_scores.append(None)

    merged["clipscore"]    = clipscores
    merged["context_sim"]  = context_sims
    merged["nli_entail"]   = entailments
    merged["gpt_relevance"] = gpt_scores

    # BERTScore batch
    print("Computing BERTScore...")
    P, R, F = bert_score(descriptions, article_texts,
                         lang="en", model_type="distilbert-base-uncased",
                         verbose=False)
    merged["bertscore_f"] = F.tolist()

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")

    # plot
    plot_df = merged.dropna(subset=[
        "clipscore","context_sim","nli_entail","bertscore_f","gpt_relevance",
        "sighted_overall","blv_overall"
    ])
    print(f"Rows with full data: {len(plot_df)}")

    metrics = ["clipscore","context_sim","nli_entail","bertscore_f","gpt_relevance"]
    xlabels = ["CLIPScore","Context Sim (SBERT)","NLI Entailment","BERTScore F1","GPT-4o Relevance"]
    ratings = ["sighted_overall","sighted_relevance","blv_overall","blv_relevance"]
    ylabels = ["Sighted Overall","Sighted Relevance","BLV Overall","BLV Relevance"]

    fig, axes = plt.subplots(len(ratings), len(metrics), figsize=(22, 16))
    fig.suptitle("Human Ratings vs. Automated Metrics (Human-Written Descriptions)",
                 fontsize=13, y=1.01)

    for row_i, (rating, ylabel) in enumerate(zip(ratings, ylabels)):
        for col_i, (metric, xlabel) in enumerate(zip(metrics, xlabels)):
            ax = axes[row_i][col_i]
            x  = plot_df[metric]
            y  = plot_df[rating]

            pearson_r,  pearson_p  = stats.pearsonr(x, y)
            spearman_r, spearman_p = stats.spearmanr(x, y)

            ax.scatter(x, y, alpha=0.4, s=25, color="steelblue",
                       edgecolors="white", linewidth=0.3)
            m, b   = np.polyfit(x, y, 1)
            x_line = pd.Series([x.min(), x.max()])
            ax.plot(x_line, m * x_line + b, color="tomato", linewidth=1.5)

            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(
                f"r={pearson_r:.2f} (p={pearson_p:.3f})\n"
                f"ρ={spearman_r:.2f} (p={spearman_p:.3f})",
                fontsize=8
            )
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("human_ratings_vs_metrics_v3.png", dpi=150, bbox_inches="tight")
    print("Saved plot to human_ratings_vs_metrics_v3.png")

if __name__ == "__main__":
    main()