import pandas as pd
import requests
import time
import json
import numpy as np
from rag_local_llm import rag_local_llm
import concurrent.futures
import matplotlib.pyplot as plt

BACKEND_URL = "http://localhost:8000"
N_SAMPLES = 200

def load_test_queries():
    """Загружает тест-кейсы"""
    try:
        articles_df = pd.read_parquet("graphrag_index/articles.parquet")
        communities_df = pd.read_parquet("graphrag_index/communities.parquet")
    except:
        print("GraphRAG index not found. Run GraphRAG first!")
        return []
    
    test_queries = []
    
    # 1. Local queries (title-based)
    local_sample = articles_df.sample(min(50, len(articles_df)), random_state=42)
    for _, row in local_sample.iterrows():
        test_queries.append({
            'query': row['title'][:200],
            'type': 'local_title',
            'gold_community': row['community_id'],
            'gold_topics': row.get('top_entities', []),
            'gold_text': row['abstract'][:500]
        })
    
    # 2. Community queries
    community_sample = communities_df.sample(min(50, len(communities_df)), random_state=42)
    for _, row in community_sample.iterrows():
        test_queries.append({
            'query': f"research on {row['summary'][:100]}",
            'type': 'community', 
            'gold_community': row['community_id'],
            'gold_topics': row.get('top_entities', []),
            'gold_text': row['summary']
        })
    
    # 3. Trend + Interdisciplinary (100 queries)
    trend_queries = [
        "reinforcement learning methods", "transformer architectures", 
        "generative adversarial networks", "graph neural networks",
        "federated learning techniques", "RL + transformers",
        "GANs + medical imaging", "NLP + federated learning"
    ] * 14  # ~100 queries
    
    for q in trend_queries[:100]:
        test_queries.append({
            'query': q,
            'type': 'trend',
            'gold_community': -1,
            'gold_topics': [],
            'gold_text': ''
        })
    
    print(f"Loaded {len(test_queries)} test queries")
    return test_queries[:N_SAMPLES]

def evaluate_rag_query(query_data):
    """TRACe RAG evaluation"""
    query = query_data['query']
    
    try:
        # Search test
        start = time.time()
        search_resp = requests.post(f"{BACKEND_URL}/query", 
                                  json={"question": query, "k": 10}, timeout=15)
        if search_resp.status_code != 200:
            raise Exception(f"Search failed: {search_resp.status_code}")
        search_resp = search_resp.json()
        search_time = time.time() - start
        
        # Pipeline test
        pipeline_start = time.time()
        summary_resp = requests.post(f"{BACKEND_URL}/summarize", 
                                   json={"question": query, "top_k": 10}, timeout=45)
        if summary_resp.status_code != 200:
            raise Exception(f"Summary failed: {summary_resp.status_code}")
        summary_resp = summary_resp.json()
        total_time = time.time() - pipeline_start
        
        # TRACe Metrics
        retrieved_communities = [s['id'] for s in search_resp.get('sources', [])]
        retrieved_topics = []
        for source in search_resp.get('sources', [])[:5]:
            retrieved_topics.extend(source.get('entities', []))
            
        gold_topics = query_data.get('gold_topics', [])
        topic_jaccard = len(set(retrieved_topics) & set(gold_topics)) / \
                       len(set(retrieved_topics) | set(gold_topics)) if retrieved_topics or gold_topics else 0
        
        summary = summary_resp.get('summary', '')
        has_structure = 1.0 if all(f"{i}." in summary for i in ['1', '2', '3']) else 0.0
        
        return {
            'query_id': hash(query) % 10000,
            'query_type': query_data['type'],
            'query_preview': query[:80],
            'search_latency_ms': round(search_time * 1000, 1),
            'pipeline_latency_ms': round((search_time + total_time) * 1000, 1),
            'retrieved_communities': len(search_resp.get('sources', [])),
            'top_score': search_resp['sources'][0]['score'] if search_resp.get('sources') else 0,
            'gold_community_recall': 1.0 if query_data['gold_community'] in retrieved_communities[:5] else 0.0,
            'summary_words': summary_resp.get('word_count', 0),
            'communities_used': summary_resp.get('top_communities', 0),
            'ollama_success': 1.0 if summary_resp.get('ollama_used') == 'phi3:mini' else 0.0,
            'has_structure': has_structure,
            'topic_jaccard': topic_jaccard
        }
        
    except Exception as e:
        return {
            'query_id': hash(query) % 10000,
            'error': str(e)[:50],
            'search_latency_ms': -1,
            'pipeline_latency_ms': -1
        }

def run_full_benchmark():
    print("Automated RAG Benchmark Test")
    print("=" * 70)
    
    test_queries = load_test_queries()
    if not test_queries:
        print("No test data. Run GraphRAG first!")
        return
    
    print(f"Testing {len(test_queries)} queries (8 workers)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(evaluate_rag_query, test_queries))
    
    df = pd.DataFrame(results)
    df = df[df['pipeline_latency_ms'] > 0]  # Filter errors
    
    if len(df) == 0:
        print("No successful tests!")
        return
    
    # Metrics
    metrics = {
        'test_date': pd.Timestamp.now().isoformat(),
        'n_queries': len(df),
        'success_rate': f"{len(df)/len(results)*100:.1f}%",
        
        'latency_p50_search': df['search_latency_ms'].quantile(0.5),
        'latency_p95_search': df['search_latency_ms'].quantile(0.95),
        'latency_p50_pipeline': df['pipeline_latency_ms'].quantile(0.5),
        'latency_p95_pipeline': df['pipeline_latency_ms'].quantile(0.95),
        
        'community_recall': df['gold_community_recall'].mean(),
        'topic_jaccard_mean': df['topic_jaccard'].mean(),
        'summary_quality': df['has_structure'].mean(),
        'ollama_success_rate': df['ollama_success'].mean()
    }
    
    # Save
    df.to_csv('rag_benchmark_results.csv', index=False)
    with open('rag_benchmark_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print_results(metrics, df)
    plot_simple_benchmarks(df, metrics)
    
    return metrics, df

def print_results(metrics, df):
    print("\n" + "="*70)
    print("RAG BENCHMARK RESULTS")
    print("="*70)
    print(f"Successful: {metrics['n_queries']} ({metrics['success_rate']})")
    print(f"Search P50: {metrics['latency_p50_search']:.0f}ms | P95: {metrics['latency_p95_search']:.0f}ms")
    print(f"Pipeline P50: {metrics['latency_p50_pipeline']:.0f}ms | P95: {metrics['latency_p95_pipeline']:.0f}ms")
    print(f"Community Recall: {metrics['community_recall']:.1%}")
    print(f"Topic Jaccard: {metrics['topic_jaccard_mean']:.3f}")
    print(f"Structured: {metrics['summary_quality']:.1%}")
    print(f"Ollama: {metrics['ollama_success_rate']:.1%}")

def plot_simple_benchmarks(df, metrics):
    """Простые графики БЕЗ seaborn"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Latency histograms
    axes[0,0].hist(df['search_latency_ms'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0,0].axvline(metrics['latency_p95_search'], color='red', ls='--', lw=2)
    axes[0,0].set_title('Search Latency (P95 line)')
    axes[0,0].set_xlabel('ms')
    
    axes[0,1].hist(df['pipeline_latency_ms'], bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0,1].axvline(metrics['latency_p95_pipeline'], color='red', ls='--', lw=2)
    axes[0,1].set_title('Pipeline Latency (P95 line)')
    axes[0,1].set_xlabel('ms')
    
    # Pie charts
    axes[1,0].pie(df['has_structure'].value_counts(), 
                  labels=['Structured', 'Unstructured'], autopct='%1.1f%%')
    axes[1,0].set_title('Summary Structure')
    
    axes[1,1].pie(df['ollama_success'].value_counts(), 
                  labels=['Phi-3', 'Fallback'], autopct='%1.1f%%', colors=['green', 'orange'])
    axes[1,1].set_title('Ollama Success')
    
    plt.tight_layout()
    plt.savefig('rag_benchmark_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plots saved: rag_benchmark_plots.png")

if __name__ == "__main__":
    metrics, df = run_full_benchmark()
    print("\FULL RAG BENCHMARK COMPLETE!")
    print("Results: rag_benchmark_results.csv")
    print("Metrics: rag_benchmark_metrics.json") 
    print("Plots: rag_benchmark_plots.png")
