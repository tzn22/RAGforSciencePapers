from pydantic import BaseModel
from typing import List, Any

class QueryRequest(BaseModel):
    q: str
    top_k: int = 5

class QueryResultItem(BaseModel):
    chunk_id: str
    title: str
    abstract: str
    text: str
    score: float

class QueryResponse(BaseModel):
    query: str
    results: List[Any]
