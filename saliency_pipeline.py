import os, torch, clip, pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from scipy import stats
from ultralytics import YOLO
from bert_score import score as bert_score

IMAGES_DIR  = "metrics/clipscore/cosid_data/cosid_images"
DESC_CSV    = "behavioral_data/all_descriptions.csv"
GPT_RESULTS = "results_gpt4o.csv"  # your existing GPT-generated captions
OUTPUT_CSV  = "results_three_axes.csv"

clip_model = preprocess = sbert = yolo = None

def setup():
    global clip_model, preprocess, sbert, yolo
    clip_model, preprocess = clip.load("ViT-B/32", device="cpu")
    sbert = SentenceTransformer("all-mpnet-base-v2")
    yolo  = YOLO("yolo11n.pt")

def get_image_path(img_file):
    img_stem   = os.path.splitext(img_file)[0]
    candidates = [f for f in os.listdir(IMAGES_DIR) if f.startswith(img_stem)]
    return os.path.join(IMAGES_DIR, candidates[0]) if candidates else None

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

def saliency_alignment(image_path, caption):
    """
    Run YOLO on image, rank detected objects by bounding box area,
    take top 3 as salient, compute SBERT cosine sim with caption.
    """
    results = yolo(image_path, verbose=False)
    boxes   = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return 0.0

    # get class names and bounding box areas
    names  = results[0].names
    labels = [names[int(c)] for c in boxes.cls]
    areas  = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes.xyxy]

    # rank by area, take top 3
    ranked  = sorted(zip(areas, labels), reverse=True)
    top_labels = [label for _, label in ranked[:3]]
    salient_text = " ".join(top_labels)

    # SBERT cosine similarity between salient objects and caption
    e1, e2 = sbert.encode([salient_text, caption])
    return util.cos_sim(e1, e2).item()

def main():
    print("Setting up models...")
    setup()

    # load GPT captions (one per image-context pair)
    gpt_df = pd.read_csv(GPT_RESULTS)
    print(f"Loaded {len(gpt_df)} image-context pairs")

    # load article texts from all_descriptions to get one per image-context
    desc_df = pd.read_csv(DESC_CSV)
    article_lookup = desc_df.groupby(
        ["img_file", "context"]
    )["article_text"].first().reset_index()

    merged = gpt_df.merge(article_lookup, on=["img_file", "context"], how="left")

    clipscores   = []
    saliencies   = []
    descriptions = []
    article_texts = []

    for i, row in merged.iterrows():
        img_path = get_image_path(row["img_file"])
        if i % 10 == 0:
            print(f"  Row {i}/{len(merged)}...")
        if img_path is None:
            clipscores.append(None)
            saliencies.append(None)
            descriptions.append(row["generated_caption"])
            article_texts.append(row.get("article_text", ""))
            continue

        clipscores.append(clipscore(img_path, row["generated_caption"]))
        saliencies.append(saliency_alignment(img_path, row["generated_caption"]))
        descriptions.append(row["generated_caption"])
        article_texts.append(row.get("article_text", ""))

    merged["clipscore"]  = clipscores
    merged["saliency"]   = saliencies

    # BERTScore recall between caption and article text
    print("Computing BERTScore recall...")
    _, R, _ = bert_score(descriptions, article_texts,
                         lang="en", model_type="distilbert-base-uncased",
                         verbose=False)
    merged["bertscore_r"] = R.tolist()

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")

    # plot three pairwise scatter plots
    plot_df = merged.dropna(subset=["clipscore", "saliency", "bertscore_r"])
    print(f"Plotting {len(plot_df)} rows")

    pairs = [
        ("clipscore",   "bertscore_r", "CLIPScore",    "BERTScore Recall"),
        ("clipscore",   "saliency",    "CLIPScore",    "Saliency Alignment"),
        ("bertscore_r", "saliency",    "BERTScore Recall", "Saliency Alignment"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Three-Axis Decomposition of Image Description Quality\n(GPT-4o Generated Captions, COSID Dataset)",
                 fontsize=13)

    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, pairs):
        x = plot_df[xcol]
        y = plot_df[ycol]

        pearson_r,  pearson_p  = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        ax.scatter(x, y, alpha=0.6, s=50, color="steelblue",
                   edgecolors="white", linewidth=0.5)
        m, b   = np.polyfit(x, y, 1)
        x_line = np.array([x.min(), x.max()])
        ax.plot(x_line, m * x_line + b, color="tomato", linewidth=2)

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(
            f"r={pearson_r:.2f} (p={pearson_p:.3f})\n"
            f"ρ={spearman_r:.2f} (p={spearman_p:.3f})",
            fontsize=10
        )
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("three_axes.png", dpi=150, bbox_inches="tight")
    print("Saved to three_axes.png")

if __name__ == "__main__":
    main()