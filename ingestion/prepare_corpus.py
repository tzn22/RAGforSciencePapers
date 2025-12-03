import os
import argparse
import uuid
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import faiss
from tqdm.auto import tqdm
import gc

INDEX_DIR = "data_index"
def chunk_text_generator(full_text, chunk_size=1500, overlap=200):
    """ГЕНЕРАТОР - НЕ хранит chunks в памяти"""
    text = full_text.replace("\n", " ").strip()
    if len(text) <= chunk_size:
        yield text
        return

    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        yield chunk  # yield вместо append
        start = end - overlap
        if start < 0:
            start = 0

def chunk_text(text, chunk_size=1500, overlap=200):
    """Simple sliding window chunking."""
    text = text.replace("\n", " ").strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0

    return chunks

def main(args):
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("Loading dataset bakhitovd/ML_arxiv...")
    ds = load_dataset("bakhitovd/ML_arxiv", split="train")

    if args.sample and args.sample < len(ds):
        ds = ds.select(range(args.sample))
        print("Using sample =", len(ds))

    print("Loaded dataset. Example fields:", ds.column_names)

    embedder = SentenceTransformer(args.embed_model)

    texts = []
    metas = []

    # ✅ ИСПРАВЛЕНИЕ: total=len(ds) для tqdm
    for i, item in enumerate(tqdm(ds, total=len(ds), desc="Processing articles")):
        title = item.get("article", "")[:200]  # article как title
        summary = item.get("abstract", "")
        full_text = item.get("article", "")

        if not isinstance(full_text, str) or len(full_text) < 100:
            continue

        for ch in chunk_text_generator(full_text, args.chunk_size, args.overlap):
            if len(ch) < 200:
                continue

            cid = str(uuid.uuid4())
            texts.append(ch)
            metas.append({
                "chunk_id": cid,
                "title": title,
                "summary": summary,
                "source_idx": i
            })

        # ✅ Очистка памяти каждые 50 статей
        if (i + 1) % 50 == 0:
            gc.collect()

    print("Encoding embeddings...")
    embs = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embs = embs.astype("float32")

    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, f"{INDEX_DIR}/faiss.index")
    pd.DataFrame(metas).assign(text=texts).to_parquet(f"{INDEX_DIR}/meta.parquet", index=False)
    np.savez_compressed(f"{INDEX_DIR}/embeddings.npz", emb=embs)

    print("Done! Index saved to data_index/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--chunk_size", type=int, default=1500)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    main(args)
