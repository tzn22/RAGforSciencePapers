# embeddings/embed_manager.py
from sentence_transformers import SentenceTransformer
import numpy as np

_model_cache = {}

def load_embedder(name="sentence-transformers/all-MiniLM-L6-v2"):
    if name not in _model_cache:
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]

def embed_texts(texts, model_name=None, batch_size=32):
    model = load_embedder(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return np.array(embs, dtype="float32")
