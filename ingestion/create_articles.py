# scripts/create_articles.py - Создать articles.parquet
from datasets import load_dataset
import pandas as pd
import os

print("📚 Создание articles.parquet...")
ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train[:100000]")
articles = []

for i, row in enumerate(ds):
    articles.append({
        "id": i,
        "title": row.get("title", ""),
        "authors": ", ".join(row.get("authors", [])),
        "abstract": row.get("abstract", ""),
        "community_id": i % 5000  # Распределяем по 20 сообществам
    })

df = pd.DataFrame(articles)
os.makedirs("graphrag_index", exist_ok=True)
df.to_parquet("graphrag_index/articles.parquet", index=False)
print(f"✅ {len(articles)} статей сохранено!")
