import os, sys, torch, clip, pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from sentence_transformers import SentenceTransformer, util
from ultralytics import YOLO
from scipy import stats
from bert_score import score as bert_score

sys.path.insert(0, os.path.expanduser('~/TranSalNet'))
from TranSalNet_Res import TranSalNet

IMAGES_DIR  = "metrics/clipscore/cosid_data/cosid_images"
GPT_RESULTS = "results_gpt4o.csv"
DESC_CSV    = "behavioral_data/all_descriptions.csv"
OUTPUT_CSV  = "results_three_axes_v2.csv"

clip_model = preprocess = sbert = yolo = sal_model = None

def setup():
    global clip_model, preprocess, sbert, yolo, sal_model
    clip_model, preprocess = clip.load("ViT-B/32", device="cpu")
    sbert = SentenceTransformer("all-mpnet-base-v2")
    yolo  = YOLO("yolov8x-oiv7.pt")
    
    # TranSalNet needs to run from its own directory
    original_dir = os.getcwd()
    os.chdir(os.path.expanduser('~/TranSalNet'))
    sal_model = TranSalNet()
    sal_model.load_state_dict(torch.load(
        'pretrained_models/TranSalNet_Res.pth',
        map_location='cpu'))
    sal_model.eval()
    os.chdir(original_dir)

def get_image_path(img_file):
    img_stem   = os.path.splitext(img_file)[0]
    candidates = [f for f in os.listdir(IMAGES_DIR)
                  if f.startswith(img_stem) and not f.startswith('.')]
    return os.path.join(IMAGES_DIR, candidates[0]) if candidates else None

def get_saliency_map(image_path):
    img = Image.open(image_path).convert('RGB').resize((384, 288))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    inp = transform(img).unsqueeze(0)
    with torch.no_grad():
        sal = sal_model(inp)
    sal_np = sal.squeeze().numpy()
    # normalize to 0-1
    sal_np = (sal_np - sal_np.min()) / (sal_np.max() - sal_np.min() + 1e-8)
    return sal_np  # shape (288, 384)

def get_salient_labels(image_path, sal_map, top_n=3):
    results = yolo(image_path, verbose=False)
    boxes   = results[0].boxes
    names   = results[0].names

    if boxes is None or len(boxes) == 0:
        return None

    h, w = sal_map.shape  # 288, 384
    img_w, img_h = Image.open(image_path).size

    scored = []
    for box, cls in zip(boxes.xyxy, boxes.cls):
        x1, y1, x2, y2 = box.tolist()
        # scale box to saliency map dimensions
        sx1 = int(x1 / img_w * w)
        sy1 = int(y1 / img_h * h)
        sx2 = int(x2 / img_w * w)
        sy2 = int(y2 / img_h * h)
        sx1, sy1 = max(0, sx1), max(0, sy1)
        sx2, sy2 = min(w, sx2), min(h, sy2)
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        region_sal = sal_map[sy1:sy2, sx1:sx2].mean()
        label = names[int(cls)]
        scored.append((region_sal, label))

    if not scored:
        return None

    scored.sort(reverse=True)
    return " ".join([label for _, label in scored[:top_n]])

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

def saliency_alignment(salient_labels, caption):
    e1, e2 = sbert.encode([salient_labels, caption])
    return util.cos_sim(e1, e2).item()

def main():
    print("Setting up models...")
    setup()

    gpt_df  = pd.read_csv(GPT_RESULTS)
    desc_df = pd.read_csv(DESC_CSV)
    article_lookup = desc_df.groupby(
        ["img_file", "context"]
    )["article_text"].first().reset_index()
    merged = gpt_df.merge(article_lookup, on=["img_file", "context"], how="left")
    print(f"Loaded {len(merged)} image-context pairs")

    clipscores    = []
    saliencies    = []
    descriptions  = []
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

        # CLIPScore
        cs = clipscore(img_path, row["generated_caption"])
        clipscores.append(cs)

        # saliency alignment
        sal_map = get_saliency_map(img_path)
        labels  = get_salient_labels(img_path, sal_map)
        if labels:
            sal_score = saliency_alignment(labels, row["generated_caption"])
        else:
            sal_score = None
        saliencies.append(sal_score)

        descriptions.append(row["generated_caption"])
        article_texts.append(row.get("article_text", ""))

    merged["clipscore"] = clipscores
    merged["saliency"]  = saliencies

    # BERTScore recall
    print("Computing BERTScore recall...")
    _, R, _ = bert_score(descriptions, article_texts,
                         lang="en", model_type="distilbert-base-uncased",
                         verbose=False)
    merged["bertscore_r"] = R.tolist()
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")

    # plot
    plot_df = merged.dropna(subset=["clipscore", "saliency", "bertscore_r"])
    print(f"Plotting {len(plot_df)} rows with full saliency data")

    pairs = [
        ("clipscore",   "bertscore_r", "CLIPScore",         "BERTScore Recall"),
        ("clipscore",   "saliency",    "CLIPScore",         "Saliency Alignment"),
        ("bertscore_r", "saliency",    "BERTScore Recall",  "Saliency Alignment"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Three-Axis Decomposition of Image Description Quality\n"
        f"(GPT-4o Captions, COSID Dataset, n={len(plot_df)})",
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
    plt.savefig("three_axes_v2.png", dpi=150, bbox_inches="tight")
    print("Saved to three_axes_v2.png")

if __name__ == "__main__":
    main()