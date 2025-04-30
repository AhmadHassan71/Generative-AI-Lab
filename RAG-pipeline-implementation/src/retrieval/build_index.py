# src/retrieval/build_index.py
import numpy as np, faiss, json
from pathlib import Path

def build_index(text_npz, img_npz, chunks_t, chunks_i, out_index, out_meta):
    # load
    t = np.load(text_npz)["vectors"]    # shape (n_text, d_t)
    i = np.load(img_npz)["vectors"]     # shape (n_img, d_i)

    # pad the smaller-dim set to match the larger
    d_t = t.shape[1]
    d_i = i.shape[1]
    if d_t > d_i:
        pad_width = ((0,0), (0, d_t - d_i))
        i = np.pad(i, pad_width, mode="constant")
    elif d_i > d_t:
        pad_width = ((0,0), (0, d_i - d_t))
        t = np.pad(t, pad_width, mode="constant")

    # now they share the same embedding dimension
    all_emb = np.vstack([t, i]).astype("float32")
    faiss.normalize_L2(all_emb)

    idx = faiss.IndexFlatIP(all_emb.shape[1])
    idx.add(all_emb)

    # write the index
    Path(out_index).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, out_index)

    # merge metadata
    with open(chunks_t) as f: meta_t = json.load(f)
    with open(chunks_i) as f: meta_i = json.load(f)
    meta = meta_t + meta_i
    Path(out_meta).parent.mkdir(parents=True, exist_ok=True)
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Built FAISS index ({all_emb.shape[0]} vectors of dim {all_emb.shape[1]})")
    print(f"Index saved to {out_index}")
    print(f"Metadata saved to {out_meta}")

if __name__=="__main__":
    import sys
    if len(sys.argv) != 7:
        print("Usage: python build_index.py <text.npz> <img.npz> <text_chunks.json> <img_chunks.json> <out.faiss> <out_meta.json>")
    else:
        build_index(*sys.argv[1:])
