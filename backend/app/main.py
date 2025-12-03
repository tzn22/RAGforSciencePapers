# backend/app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, time
from retrieval.retriever import Retriever
from embeddings.embed_manager import load_embedder
from reranker.cross_rerank import rerank
from backend.app.rag_local_llm import load_llm, llm_summarize

app = FastAPI(title="Local RAG Backend")
RETRIEVER = None
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")

class QueryReq(BaseModel):
    q: str
    top_k: int = 5

@app.on_event("startup")
def startup():
    global RETRIEVER
    print("Starting Retriever...")
    RETRIEVER = Retriever()
    print("Loading embedder...")
    # load embedder into cache
    load_embedder(EMBED_MODEL)
    # load LLM lazily

@app.post("/query")
def query(req: QueryReq):
    if not req.q.strip():
        raise HTTPException(400, "Empty query")
    embedder = load_embedder(EMBED_MODEL)
    start = time.time()
    cand_idxs = RETRIEVER.hybrid(embedder, req.q, top_k=200, bm25_k=50)
    candidate_texts = [RETRIEVER.meta.iloc[i]["text"] for i in cand_idxs]
    r_idx, scores = rerank(req.q, candidate_texts, model_name=RERANK_MODEL, top_k=req.top_k)
    results = []
    for rel_idx, sc in zip(r_idx, scores):
        idx = cand_idxs[rel_idx]
        row = RETRIEVER.meta.iloc[idx].to_dict()
        row["score"] = float(sc)
        results.append(row)
    latency = time.time() - start
    return {"query": req.q, "results": results, "latency": latency}

class SummReq(BaseModel):
    q: str
    top_k: int = 5

@app.post("/summarize")
def summarize(req: SummReq):
    # get retrieval results
    qr = query(QueryReq(q=req.q, top_k=req.top_k))
    passages = [r["text"] for r in qr["results"]]
    # load llm
    tokenizer, model = load_llm(LLM_MODEL)
    summary = llm_summarize(req.q, passages, tokenizer, model)
    return {"query": req.q, "summary": summary, "retrieval": qr}
