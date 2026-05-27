import os, base64, torch, clip, pandas as pd
from openai import OpenAI
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

IMAGES_DIR = "metrics/clipscore/cosid_data/cosid_images"
DESC_CSV   = "behavioral_data/all_descriptions.csv"
OUTPUT_CSV = "results_gpt4o.csv"

client = clip_model = preprocess = sbert = None

def setup():
    global client, clip_model, preprocess, sbert
    client = OpenAI(api_key=OPENAI_KEY)
    clip_model, preprocess = clip.load("ViT-B/32", device="cpu")
    sbert = SentenceTransformer("all-mpnet-base-v2")

def generate_caption(image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": "Describe this image in one or two sentences as an alt text description for a visually impaired user."}
        ]}],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

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

def main():
    print("Starting pipeline...")
    setup()
    print("Setup complete.")

    df = pd.read_csv(DESC_CSV)
    unique_images = df[["img_id", "img_file"]].drop_duplicates(subset="img_file")
    print(f"Found {len(unique_images)} unique images.")

    # generate one caption per unique image
    caption_cache = {}
    for _, row in unique_images.iterrows():
        img_file = row["img_file"]
        img_stem = os.path.splitext(img_file)[0]
        candidates = [f for f in os.listdir(IMAGES_DIR) if f.startswith(img_stem)]
        if not candidates:
            print(f"MISSING: {img_stem[:50]}, skipping")
            continue
        img_path = os.path.join(IMAGES_DIR, candidates[0])
        print(f"Generating caption for {img_stem[:50]}...")
        try:
            caption_cache[img_file] = generate_caption(img_path)
            print(f"  -> {caption_cache[img_file][:80]}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # for each image-context pair, compute scores
    results = []
    seen = set()
    for _, row in df.iterrows():
        img_file = row["img_file"]
        context  = row["context"]
        key      = (img_file, context)
        if key in seen or img_file not in caption_cache:
            continue
        seen.add(key)

        img_stem   = os.path.splitext(img_file)[0]
        candidates = [f for f in os.listdir(IMAGES_DIR) if f.startswith(img_stem)]
        if not candidates:
            continue
        img_path = os.path.join(IMAGES_DIR, candidates[0])
        caption  = caption_cache[img_file]
        cs       = clipscore(img_path, caption)
        sim      = context_sim(row["article_text"], caption)

        results.append({
            "img_file":          img_file,
            "context":           context,
            "generated_caption": caption,
            "clipscore_gpt4o":   cs,
            "context_sim":       sim,
        })

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Saved {len(out)} rows to {OUTPUT_CSV}")
    print(out[["context", "clipscore_gpt4o", "context_sim"]].to_string())

if __name__ == "__main__":
    main()