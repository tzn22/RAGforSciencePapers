"""
BM25 over chunks (fast approximate) using rank_bm25
"""
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import os
import pickle

# ensure punkt installed
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

class BM25Index:
    def __init__(self, docs=None):
        self.docs = docs or []
        self.tokenized = [word_tokenize(d.lower()) for d in self.docs]
        self.index = BM25Okapi(self.tokenized)

    def add_docs(self, docs):
        self.docs.extend(docs)
        self.tokenized = [word_tokenize(d.lower()) for d in self.docs]
        self.index = BM25Okapi(self.tokenized)

    def search(self, query, top_k=10):
        q_tok = word_tokenize(query.lower())
        scores = self.index.get_scores(q_tok)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:top_k]
