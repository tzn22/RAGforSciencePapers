# backend/app/main.py - ✅ МАСШТАБИРОВАННАЯ версия (100K ready)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import requests
import pandas as pd
from backend.app.rag_local_llm import rag_local_llm 

app = FastAPI(
    title="Scientific Literature Review Platform",
    description="Knowledge Graph RAG for ML/AI literature discovery",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    k: int = 10

class SummarizeRequest(BaseModel):
    question: str
    top_k: int = 10
@app.get("/")
async def root():
    return {"message": "Scientific Literature Review Platform ready"}

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Search ML/AI literature communities"""
    start = time.time()
    result = rag_local_llm(request.question, request.k)
    result["latency_ms"] = round((time.time() - start) * 1000, 2)
    return result

@app.post("/summarize")
async def summarize_endpoint(request: SummarizeRequest):
    """Generate professional literature review"""
    start = time.time()
    
    retrieval_start = time.time()
    rag_result = rag_local_llm(request.question, request.top_k)
    retrieval_latency = round((time.time() - retrieval_start) * 1000, 2)
    
    sources = rag_result["sources"]
    passages = [s["summary"] for s in sources[:5]]
    
    try:
        health = requests.get("http://localhost:11434/api/tags", timeout=5)
        if health.status_code != 200:
            raise HTTPException(status_code=503, detail="Ollama service unavailable")
        
        models = health.json().get("models", [])
        if not any("phi3" in m.get("name", "").lower() for m in models):
            raise HTTPException(status_code=503, detail="Phi-3 model not found")
        context = "\n".join([f"[{i+1}] {p[:300]}" for i, p in enumerate(passages)])
        prompt = f"""Generate a professional literature review:

CONTEXT (Top 5 ML/AI research communities):
{context}

**Required Structure:**
**1. Main Topics:** One sentence summarizing research focus
**2. Key Methods:** One sentence listing core techniques  
**3. Applications:** One sentence describing real-world impact

Rules: Complete sentences. Technical terminology. No truncation. Concise."""

        ollama_resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3:mini",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 512,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "stop": ["\n\n", "###"] 
                }
            },
            timeout=30
        ).json()
        
        summary = ollama_resp.get("response", "").strip()
        words = summary.split()
        if len(words) > 300:
            summary = " ".join(words[:300]) + "..."
            
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Ollama timeout")
    except Exception as e:
        print(f"⚠️ Ollama error: {e}")
        
        key_topics = []
        for p in passages[:5]:
            if '**' in p:
                topic = p.split('**')[1].split(',')[0].strip()
            else:
                topic = p.split(',')[0].strip()
            key_topics.append(topic)
        
        summary = f"""**1. Main Topics:** Research communities focused on {', '.join(key_topics[:5])} in machine learning and AI.

**2. Key Methods:** Knowledge graph clustering, TF-IDF semantic search, community detection algorithms.

**3. Applications:** Scientific literature discovery, interdisciplinary research mapping, emerging trend identification."""

    total_latency = (time.time() - start) * 1000
    generation_latency = round(total_latency - retrieval_latency, 0)
    
    return {
        "question": request.question,
        "summary": summary,
        "retrieval_latency": retrieval_latency,
        "generation_latency": generation_latency,
        "word_count": len(summary.split()),
        "char_count": len(summary),
        "context": passages,
        "n_sources": len(sources),
        "top_communities": len(passages),
        "ollama_used": "phi3:mini" if "ollama_resp" in locals() else "fallback"
    }

@app.get("/health")
async def health_check():
    """System health check"""
    try:
        ollama_resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        ollama_ok = ollama_resp.status_code == 200
    except:
        ollama_ok = False
    
    try:
        communities = pd.read_parquet("graphrag_index/communities.parquet")
        rag_ok = len(communities) > 0
    except:
        rag_ok = False
    
    return {
        "status": "healthy" if ollama_ok and rag_ok else "degraded",
        "ollama": ollama_ok,
        "rag_index": rag_ok,
        "timestamp": time.time()
    }

@app.get("/ollama-status")
async def ollama_status():
    """Ollama diagnostics"""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = resp.json().get("models", [])
        phi3_available = any("phi3" in m.get("name", "").lower() for m in models)
        return {
            "status": "ok", 
            "phi3_available": phi3_available,
            "total_models": len(models),
            "models": [m.get("name") for m in models[:5]]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/debug")
async def debug():
    """Index statistics"""
    communities = pd.read_parquet("graphrag_index/communities.parquet")
    try:
        articles = pd.read_parquet("graphrag_index/articles.parquet")
        return {
            "communities": len(communities),
            "articles": len(articles),
            "avg_articles_per_community": len(articles) / max(len(communities), 1)
        }
    except:
        return {"communities": len(communities), "articles": 0}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
