# retrieval/retriever.py
import os, faiss, numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from embeddings.embed_manager import load_embedder

INDEX_DIR = os.getenv("INDEX_DIR", "data_index")

class Retriever:
    def __init__(self, index_dir=INDEX_DIR):
        self.index_dir = index_dir
        idx_path = os.path.join(index_dir, "faiss.index")
        meta_path = os.path.join(index_dir, "meta.parquet")
        emb_path = os.path.join(index_dir, "embeddings.npz")
        if not os.path.exists(idx_path):
            raise FileNotFoundError("FAISS index not found; run ingestion first.")
        self.index = faiss.read_index(idx_path)
        self.meta = pd.read_parquet(meta_path)
        self.embs = np.load(emb_path)["emb"]
        self.bm25 = BM25Okapi([word_tokenize(t.lower()) for t in self.meta["text"].astype(str).tolist()])

    def vector_search(self, model, query, top_k=50):
        q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        return D[0], I[0].tolist()

    def bm25_search(self, query, top_k=20):
        q_tok = word_tokenize(query.lower())
        scores = self.bm25.get_scores(q_tok)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:top_k]

    def hybrid(self, model, query, top_k=50, bm25_k=20):
        bm = self.bm25_search(query, top_k=bm25_k)
        _, vec_idx = self.vector_search(model, query, top_k=top_k)
        merged = []
        seen = set()
        for i in bm + list(vec_idx):
            if int(i) not in seen:
                merged.append(int(i)); seen.add(int(i))
        return merged[:top_k]
