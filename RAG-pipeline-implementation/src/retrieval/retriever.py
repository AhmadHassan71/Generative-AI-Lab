# src/retrieval/retriever.py
import faiss, json, numpy as np
from pathlib import Path

class Retriever:
    def __init__(self, index_file, meta_file):
        self.index = faiss.read_index(index_file)
        self.meta = json.load(open(meta_file))
    def query(self, q_vec, top_k=5):
        faiss.normalize_L2(q_vec)
        D, I = self.index.search(q_vec, top_k)
        out = []
        for dist, idx in zip(D[0], I[0]):
            m = self.meta[idx].copy()
            m["score"] = float(dist)
            out.append(m)
        return out

if __name__=="__main__":
    import sys, numpy as np
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-mpnet-base-v2")
    retr = Retriever(sys.argv[1], sys.argv[2])
    q = "What are the cash flow components?"
    v = model.encode([q], normalize_embeddings=True).astype("float32")
    print(retr.query(v, 5))
