# backend/app/rag_local_llm.py - GraphRAG (unchanged)
import os
import pandas as pd
import numpy as np
import faiss
import re
from sklearn.feature_extraction.text import TfidfVectorizer

GRAPH_INDEX_DIR = "graphrag_index"
_retriever = None

class InstantRetriever:
    def __init__(self):
        global _retriever
        if _retriever:
            self.__dict__ = _retriever.__dict__
            return
            
        print("🔍 Loading GraphRAG index...")
        
        self.communities = pd.read_parquet(f"{GRAPH_INDEX_DIR}/communities.parquet")
        summaries = self.communities["summary"].fillna("empty").astype(str).str[:1000]
        
        self.tfidf = TfidfVectorizer(
            max_features=2000, ngram_range=(1,2), 
            stop_words="english", lowercase=True
        )
        self.tfidf.fit(summaries)
        
        corpus_embs = self.tfidf.transform(summaries).toarray().astype(np.float32)
        corpus_embs = np.ascontiguousarray(corpus_embs)
        faiss.normalize_L2(corpus_embs)
        
        self.index = faiss.IndexFlatIP(corpus_embs.shape[1])
        self.index.add(corpus_embs)
        
        articles_path = f"{GRAPH_INDEX_DIR}/articles.parquet"
        self.articles_df = pd.read_parquet(articles_path) if os.path.exists(articles_path) else pd.DataFrame()
        
        self.community_articles = {}
        for cid in self.communities['community_id'].unique():
            if len(self.articles_df) > 0 and 'community_id' in self.articles_df.columns:
                mask = self.articles_df['community_id'] == cid
                self.community_articles[cid] = self.articles_df[mask].to_dict('records')
            else:
                self.community_articles[cid] = []
        
        _retriever = self
        print(f"✅ {self.index.ntotal} communities | 📚 {len(self.articles_df)} articles")

def clean_summary(raw_summary):
    raw_summary = str(raw_summary)
    if "This community consists of" in raw_summary or len(raw_summary.split()) > 50:
        terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)?\b', raw_summary)
        clean_terms = [t for t in terms if 3 < len(t) < 25 and t.lower() not in ['Pdf', 'Us']]
        if len(clean_terms) >= 3:
            return f"**{', '.join(clean_terms[:8])}** - ML/AI research ({len(clean_terms)} topics)"
        return "General ML/AI research cluster"
    return raw_summary[:400]

retriever = InstantRetriever()

def graphrag_query(query, k=5):
    try:
        q_emb = retriever.tfidf.transform([query]).toarray().astype(np.float32)
        q_emb = np.ascontiguousarray(q_emb)
        faiss.normalize_L2(q_emb)
        
        scores, idxs = retriever.index.search(q_emb, k)
        results = []
        
        for i, idx in enumerate(idxs[0]):
            if 0 <= idx < len(retriever.communities):
                row = retriever.communities.iloc[idx]
                cid = int(row['community_id'])
                
                results.append({
                    "id": cid,
                    "score": float(scores[0][i]),
                    "summary": clean_summary(row["summary"]),
                    "entities": row.get("top_entities", []) if isinstance(row.get("top_entities"), list) else [],
                    "articles": retriever.community_articles.get(cid, [])[:5]
                })
        
        while len(results) < min(3, k):
            fallback_cid = len(results) % len(retriever.communities)
            row = retriever.communities.iloc[fallback_cid]
            results.append({
                "id": int(row['community_id']),
                "score": 0.1,
                "summary": clean_summary(row["summary"]),
                "entities": [],
                "articles": []
            })
            
        return {
            "question": query,
            "sources": results,
            "n_sources": len(results),
            "debug_info": {
                "communities_total": len(retriever.communities),
                "articles_total": len(retriever.articles_df)
            }
        }
    except Exception as e:
        print(f"❌ Query error: {e}")
        return {"question": query, "sources": [], "n_sources": 0, "error": str(e)}

def rag_local_llm(query, k=5):
    return graphrag_query(query, k)
