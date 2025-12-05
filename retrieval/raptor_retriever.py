# retrieval/raptor_retriever.py
import os
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

INDEX_ROOT = os.getenv("INDEX_ROOT", "data_index")

class RaptorRetriever:
    def __init__(self, embed_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.index_root = INDEX_ROOT
        self.embed_model = embed_model
        self.embedder = SentenceTransformer(embed_model)
        self.levels = []
        self.metas = {}       # lvl -> dataframe
        self.indexes = {}     # lvl -> faiss index
        self.embs = {}        # lvl -> np.array embeddings
        self.nodeid_to_idx = {}  # lvl -> dict node_id -> row idx
        self._load_manifest_and_indexes()

    def _load_manifest_and_indexes(self):
        man_path = os.path.join(self.index_root, "manifest.json")
        if not os.path.exists(man_path):
            raise FileNotFoundError("manifest.json not found in data_index. Run ingestion first.")
        with open(man_path, "r", encoding="utf-8") as f:
            man = json.load(f)
        num_levels = man.get("levels", 1)
        self.levels = list(range(0, num_levels))
        for lvl in self.levels:
            meta_path = os.path.join(self.index_root, f"meta_L{lvl}.parquet")
            idx_path = os.path.join(self.index_root, f"faiss_L{lvl}.index")
            emb_path = os.path.join(self.index_root, f"emb_L{lvl}.npz")
            if os.path.exists(meta_path):
                df = pd.read_parquet(meta_path)
                self.metas[lvl] = df
                # build id->idx mapping
                mapping = {row["node_id"]: i for i, row in df.reset_index().iterrows()}
                self.nodeid_to_idx[lvl] = mapping
            else:
                self.metas[lvl] = None
                self.nodeid_to_idx[lvl] = {}
            if os.path.exists(idx_path):
                self.indexes[lvl] = faiss.read_index(idx_path)
            else:
                self.indexes[lvl] = None
            if os.path.exists(emb_path):
                try:
                    self.embs[lvl] = np.load(emb_path)["arr_0"] if "arr_0" in np.load(emb_path) else np.load(emb_path)["emb"]
                except Exception:
                    self.embs[lvl] = np.load(emb_path)["emb"]
            else:
                self.embs[lvl] = None

        # build BM25 on L0 (fine texts) if available
        if self.metas.get(0) is not None:
            docs = self.metas[0]["summary"].astype(str).tolist()
            tokenized = [word_tokenize(d.lower()) for d in docs]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    def _embed_query(self, query):
        emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(emb)
        return emb

    def _vector_search(self, lvl, query_emb, top_k=10):
        idx = self.indexes.get(lvl)
        if idx is None:
            return [], []
        D, I = idx.search(query_emb, top_k)
        return D[0].tolist(), I[0].tolist()

    def _bm25_topk(self, query, top_k=50):
        if self.bm25 is None:
            return []
        toks = word_tokenize(query.lower())
        scores = self.bm25.get_scores(toks)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:top_k]

    def hierarchical_retrieve(self, query, top_k=5, expand_factor=3):
        """
        1) search highest level (Lmax)
        2) pick top candidates
        3) expand each candidate downwards (children) using vector similarity at lower level
        4) collect nodes across levels, score and deduplicate, return top_k nodes
        """
        if not self.levels:
            raise RuntimeError("No levels loaded.")
        max_lvl = max(self.levels)
        q_emb = self._embed_query(query)

        # 1) search top-level
        # if top-level index is None, fallback to next level down
        lvl = max_lvl
        while lvl>=0 and self.indexes.get(lvl) is None:
            lvl -= 1
        if lvl < 0:
            return []

        D_top, I_top = self._vector_search(lvl, q_emb, top_k=top_k*expand_factor)
        candidates = []
        for score, idx in zip(D_top, I_top):
            meta = self.metas[lvl].iloc[int(idx)].to_dict()
            meta["_level"] = lvl
            meta["_idx_in_level"] = int(idx)
            meta["_score"] = float(score)
            candidates.append(meta)

        # 2) expand each candidate downward
        expanded = []
        for cand in candidates:
            expanded.append(cand)
            parent_level = cand["_level"]
            # expand down levels
            for child_level in range(parent_level-1, -1, -1):
                # find similar nodes in child_level by vector search using candidate.summary as query
                text_for_search = cand.get("summary", "")[:2000]
                if not text_for_search:
                    continue
                q_emb2 = self.embedder.encode([text_for_search], convert_to_numpy=True).astype("float32")
                faiss.normalize_L2(q_emb2)
                D2, I2 = self._vector_search(child_level, q_emb2, top_k=expand_factor)
                for sc2, idx2 in zip(D2, I2):
                    row = self.metas[child_level].iloc[int(idx2)].to_dict()
                    row["_level"] = child_level
                    row["_idx_in_level"] = int(idx2)
                    row["_score"] = float(sc2) * 0.9  # slight downweight
                    expanded.append(row)

        # 3) combine BM25 signals for L0 if available
        bm25_candidates = set(self._bm25_topk(query, top_k=top_k*expand_factor))
        if bm25_candidates and self.metas.get(0) is not None:
            for idx in bm25_candidates:
                row = self.metas[0].iloc[int(idx)].to_dict()
                row["_level"] = 0
                row["_idx_in_level"] = int(idx)
                row["_score"] = row.get("_score", 0.0) + 0.5  # boost
                expanded.append(row)

        # 4) deduplicate by node_id and pick best score
        best_map = {}
        for e in expanded:
            nid = e.get("node_id")
            if nid is None:
                continue
            sc = e.get("_score", 0.0)
            if nid not in best_map or sc > best_map[nid]["_score"]:
                best_map[nid] = e

        final = sorted(best_map.values(), key=lambda x: x["_score"], reverse=True)
        return final[:top_k]

    def retrieve_with_tree(self, query, top_k=5, expand_factor=3):
        """
        Returns: list of nodes (dicts) ordered by score.
        Each node contains at least: node_id, level, summary, child_ids, cluster_member_ids, _score
        """
        return self.hierarchical_retrieve(query, top_k=top_k, expand_factor=expand_factor)
