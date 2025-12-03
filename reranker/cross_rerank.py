# reranker/cross_rerank.py
from sentence_transformers import CrossEncoder

_cache = {}
def get_cross_encoder(name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    if name not in _cache:
        _cache[name] = CrossEncoder(name)
    return _cache[name]

def rerank(query, candidates_texts, model_name=None, top_k=5):
    model = get_cross_encoder(model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, t] for t in candidates_texts]
    scores = model.predict(pairs)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return ranked, [float(scores[i]) for i in ranked]
