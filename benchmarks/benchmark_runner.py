# benchmarks/benchmark_runner.py
import csv, requests, time, os
from fuzzywuzzy import fuzz

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")
QA_FILE = "benchmarks/gold_qa.csv"
OUT = "benchmarks/results.csv"

def run():
    rows = []
    with open(QA_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = r["question"]; gold = r.get("answer","")
            t0 = time.time()
            res = requests.post(f"{BACKEND}/query", json={"q": q, "top_k": 10}, timeout=120).json()
            t1 = time.time()
            texts = " ".join([x.get("text","") for x in res.get("results",[])])
            best = max([fuzz.partial_ratio(gold, t) for t in [texts]]) if gold else 0
            rows.append({"id": r["id"], "question": q, "time": t1-t0, "best_fuzzy": best})
    with open(OUT, "w", newline='', encoding='utf-8') as wf:
        writer = csv.DictWriter(wf, fieldnames=["id","question","time","best_fuzzy"])
        writer.writeheader(); writer.writerows(rows)
    print("Saved", OUT)

if __name__ == "__main__":
    run()
