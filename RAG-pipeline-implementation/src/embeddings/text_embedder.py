# src/embeddings/text_embedder.py
import json, numpy as np
import os
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Set the KMP_DUPLICATE_LIB_OK environment variable to avoid OpenMP runtime conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def embed_text(chunks_json, out_npz):
    with open(chunks_json) as f:
        chunks = json.load(f)
    model = SentenceTransformer("all-mpnet-base-v2")
    texts = [c["text"] for c in chunks]
    vecs = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, vectors=vecs)
    print(f"Saved {len(vecs)} text embeddings to {out_npz}")

if __name__=="__main__":
    import sys
    embed_text(sys.argv[1], sys.argv[2])
