# src/embeddings/image_embedder.py
import json, numpy as np, torch
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
from PIL import Image

def embed_images(chunks_json, out_npz):
    with open(chunks_json) as f:
        images = json.load(f)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    vecs = []
    for img in images:
        im = Image.open(img["path"]).convert("RGB")
        inp = proc(images=im, return_tensors="pt")
        with torch.no_grad():
            feat = model.get_image_features(**inp).cpu().numpy()[0]
        vecs.append(feat/np.linalg.norm(feat))
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, vectors=np.vstack(vecs))
    print(f"Saved {len(vecs)} image embeddings to {out_npz}")

if __name__=="__main__":
    import sys
    embed_images(sys.argv[1], sys.argv[2])
