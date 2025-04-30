# src/chunking/chunker.py
import json
from pathlib import Path

def chunk_text(pages_json, out_json, max_words=300, overlap=50):
    with open(pages_json) as f:
        pages = json.load(f)
    chunks = []
    for pg in pages:
        words = pg["text"].split()
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i+max_words])
            chunks.append({"page": pg["page"], "text": chunk})
            i += max_words - overlap
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(chunks, f, indent=2)

def chunk_images(images_json, out_json):
    with open(images_json) as f:
        images = json.load(f)
    # images already have page, path, caption
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(images, f, indent=2)

if __name__ == "__main__":
    import sys
    px, ix, out_t, out_i = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    chunk_text(px, out_t)
    chunk_images(ix, out_i)
    print("Chunked text ->", out_t)
    print("Chunked images ->", out_i)
