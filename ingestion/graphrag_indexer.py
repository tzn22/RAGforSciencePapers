# ingestion/graphrag_indexer.py
import os
import json
import pickle
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------
# NLP (spaCy for entity extraction)
# ----------------------------------------------------
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    os.system("pip install spacy && python -m spacy download en_core_web_sm")
    import spacy
    nlp = spacy.load("en_core_web_sm")


# ----------------------------------------------------
# Simple TF-IDF + Random Projection Embedder
# ----------------------------------------------------
class SimpleEmbedder:
    def __init__(self, embed_dim=384, max_vocab=5000):
        self.embed_dim = embed_dim
        self.max_vocab = max_vocab
        self.vectorizer = None
        self.proj_matrix = None

    def fit(self, texts):
        print("📌 Building TF-IDF vocabulary...")

        # good settings for scientific text
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_vocab,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )

        self.vectorizer.fit(texts)
        vocab_size = len(self.vectorizer.vocabulary_)

        # random Gaussian projection
        self.proj_matrix = (
            np.random.randn(vocab_size, self.embed_dim).astype(np.float32) * 0.1
        )

        print(f"   → vocab_size = {vocab_size}")
        return vocab_size

    def encode(self, texts):
        tfidf = self.vectorizer.transform(texts).toarray().astype(np.float32)

        v_tfidf = tfidf.shape[1]
        v_proj = self.proj_matrix.shape[0]

        # pad or trim
        if v_tfidf < v_proj:
            tfidf = np.pad(tfidf, ((0, 0), (0, v_proj - v_tfidf)))
        elif v_tfidf > v_proj:
            tfidf = tfidf[:, :v_proj]

        emb = tfidf @ self.proj_matrix  # (N, D)
        emb = np.ascontiguousarray(emb, dtype=np.float32)

        # normalize
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        emb /= norms
        return np.ascontiguousarray(emb, dtype=np.float32)


# ----------------------------------------------------
# GraphRAG Indexer
# ----------------------------------------------------
class GraphRAG:
    def __init__(self, embed_dim=384):
        self.embed_dim = embed_dim
        self.graph = nx.MultiDiGraph()
        self.embedder = SimpleEmbedder(embed_dim=embed_dim, max_vocab=5000)
        self.communities = {}

    # -------------------------------
    # ENTITY + RELATION EXTRACTION
    # -------------------------------
    def extract_entities_relations(self, text):
        try:
            doc = nlp(text[:2000])
        except Exception:
            return [], []

        entities = []
        for ent in doc.ents:
            if ent.label_ in ["ORG", "GPE", "PERSON", "PRODUCT", "NORP"]:
                entities.append((ent.text, ent.label_))

        relations = []
        sents = list(doc.sents)
        for sent in sents:
            ents = [e.text for e in sent.ents]
            if len(ents) >= 2:
                for i in range(len(ents) - 1):
                    relations.append((ents[i], "RELATED_TO", ents[i + 1]))

        return entities, relations[:5]

    # -------------------------------
    # KNOWLEDGE GRAPH BUILDING
    # -------------------------------
    def build_graph(self, texts):
        print("🧠 Extracting entities and relations...")

        for doc_id, text in enumerate(tqdm(texts, desc="Extracting")):
            ents, rels = self.extract_entities_relations(text)

            for ent, label in ents:

                # --- гарантируем наличие всех атрибутов ---
                if not self.graph.has_node(ent):
                    self.graph.add_node(ent,
                                        type="entity",
                                        label=label,
                                        docs=[doc_id])
                else:
                    node = self.graph.nodes[ent]

                    # type
                    if "type" not in node:
                        node["type"] = "entity"

                    # label
                    if "label" not in node:
                        node["label"] = label

                    # docs
                    if "docs" not in node:
                        node["docs"] = [doc_id]
                    else:
                        node["docs"].append(doc_id)

            # --- relations ---
            for src, rel, tgt in rels:
                if not self.graph.has_node(src):
                    self.graph.add_node(src, type="entity", docs=[doc_id])
                if not self.graph.has_node(tgt):
                    self.graph.add_node(tgt, type="entity", docs=[doc_id])

                self.graph.add_edge(src, tgt,
                                    relation=rel,
                                    weight=1.0,
                                    doc_id=doc_id)

        print(f"✔ Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")


    # -------------------------------
    # COMMUNITY DETECTION (FAST)
    # -------------------------------
    def detect_communities(self, n=20):
        print("📌 Detecting communities...")

        degrees = dict(self.graph.degree())
        nodes_sorted = sorted(degrees, key=lambda x: degrees[x], reverse=True)

        communities = defaultdict(list)
        for i, node in enumerate(nodes_sorted):
            communities[i % n].append(node)

        self.communities = {}
        summaries = []

        for cid, nodes_list in communities.items():
            # BETTER SUMMARY: include entities + relationships + names
            summary_text = (
                " ".join(nodes_list[:20]) +
                " . This community consists of " +
                f"{len(nodes_list)} scientific entities from ML/AI papers."
            )

            self.communities[cid] = {
                "community_id": cid,
                "size": len(nodes_list),
                "summary": summary_text,
                "top_entities": nodes_list[:10]
            }
            summaries.append(summary_text)

        # embeddings
        print("📌 Embedding community summaries...")
        vocab = self.embedder.fit(summaries)
        emb = self.embedder.encode(summaries)

        # attach embeddings
        for cid, e in zip(self.communities.keys(), emb):
            self.communities[cid]["embedding"] = e

        print(f"✔ {len(self.communities)} communities created (vocab={vocab})")

    # -------------------------------
    # SAVE INDEX
    # -------------------------------
    def save(self, out_dir="graphrag_index"):
        os.makedirs(out_dir, exist_ok=True)

        # save KG
        with open(os.path.join(out_dir, "knowledge_graph.pickle"), "wb") as f:
            pickle.dump(self.graph, f)

        # save communities dataframe
        df = pd.DataFrame(list(self.communities.values()))
        df = df.reset_index(drop=True)         # CRITICAL FIX
        df.to_parquet(os.path.join(out_dir, "communities.parquet"))

        # FAISS index
        embs = np.stack([c["embedding"] for c in self.communities.values()])
        faiss.normalize_L2(embs)
        faiss_index = faiss.IndexFlatIP(self.embed_dim)
        faiss_index.add(embs.astype(np.float32))
        faiss.write_index(faiss_index, os.path.join(out_dir, "communities.faiss"))

        # save projection + vocab
        np.save(os.path.join(out_dir, "proj_matrix.npy"), self.embedder.proj_matrix)

        # vocabulary must be JSON serializable
        vocab = {str(k): int(v) for k, v in self.embedder.vectorizer.vocabulary_.items()}
        with open(os.path.join(out_dir, "vocabulary.json"), "w") as f:
            json.dump(vocab, f)

        # manifest
        manifest = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "communities": len(self.communities),
            "embed_dim": self.embed_dim
        }
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"🎉 GraphRAG index saved: {out_dir}/")


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main(args):
    print("📦 Loading dataset...")

    ds = load_dataset(args.dataset, split="train")
    if args.sample:
        ds = ds.select(range(min(args.sample, len(ds))))

    texts = []
    for row in ds:
        title = row.get("title", "")
        abstract = row.get("abstract", row.get("summary", ""))
        txt = f"{title}. {abstract}".strip()
        if len(txt) > 70:
            texts.append(txt)

    print(f"✔ Loaded {len(texts)} documents")

    rag = GraphRAG(embed_dim=384)
    rag.build_graph(texts)
    rag.detect_communities(args.n_communities)
    rag.save(args.output)
    print("🎯 DONE!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CShorten/ML-ArXiv-Papers")
    parser.add_argument("--sample", type=int, default=50000)
    parser.add_argument("--n_communities", type=int, default=20)
    parser.add_argument("--output", default="graphrag_index")
    args = parser.parse_args()
    main(args)
