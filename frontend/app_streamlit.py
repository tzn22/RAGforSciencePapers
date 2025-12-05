# frontend/app.py - v2.3: Shows communities used!
import streamlit as st
import requests
import os
from dotenv import load_dotenv
load_dotenv()

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Scientific Literature Review Platform", page_icon="🔬", layout="wide")
st.title("🔬 Scientific Literature Review and Discovery Platform")
st.markdown("**Knowledge Graph RAG for ML/AI scientific literature**")

if "results" not in st.session_state:
    st.session_state.results = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "query_text" not in st.session_state:
    st.session_state.query_text = ""

with st.sidebar:
    st.header("⚙️ Search Settings")
    k = st.slider("Top K communities", 1, 10, 5)


q = st.text_input("🔍 Search scientific literature", value=st.session_state.query_text)

col1, col2 = st.columns([3, 1])
with col1:
    if st.button("🚀 Search Literature", type="primary", use_container_width=True):
        if q.strip():
            st.session_state.query_text = q
            with st.spinner("Searching Knowledge Graph..."):
                resp = requests.post(f"{BACKEND}/query", json={"question": q, "k": k}, timeout=30).json()
                st.session_state.results = resp
                n_sources = resp.get('n_sources', 0)
                st.success(f"✅ Found **{n_sources}** communities ({resp.get('latency_ms', 0):.0f}ms)")

if st.session_state.results:
    r = st.session_state.results
    col1, col2, col3 = st.columns([1,1,1])
    col1.metric("⏱️ Search Time", f"{r.get('latency_ms', 0):.0f}ms")
    col2.metric("📈 Communities", r.get('n_sources', 0))
    col3.metric("📚 Total Articles", r.get('debug_info', {}).get('articles_total', 0))
    
    st.subheader("🔗 Found Knowledge Graph Communities")
    
    for i, node in enumerate(r.get("sources", []), 1):
        st.markdown("---")
        with st.expander(f"**{i}.** Community #{node.get('id')} (score: {node.get('score', 0):.3f})"):
            st.markdown(f"**📄 Tags:** {node.get('summary', '')}")
            
            articles = node.get('articles', [])
            if articles:
                st.markdown(f"**📚 Articles ({len(articles)}):**")
                cols = st.columns(2)
                for j, article in enumerate(articles[:10]):
                    with cols[j % 2]:
                        title = article.get('title', f'Article {j+1}')
                        st.markdown(f"**{title}**")
                        authors = article.get('authors', 'N/A')
                        if authors:
                            st.caption(f"*{authors}*")
                        abstract = article.get('abstract', '')
                        if abstract:
                            st.markdown("**Abstract:**")
                            st.write(abstract)
            
            entities = node.get("entities", [])
            if entities:
                st.markdown(f"**🏷️ Key Entities:** {', '.join(entities)}")
    
    if st.button("✨ Generate Summary", type="secondary", use_container_width=True):
        with st.spinner("Generating insights ..."):
            try:
                s = requests.post(f"{BACKEND}/summarize", json={"question": q, "top_k": 3}, timeout=30).json()
                st.session_state.summary = s
                st.success("✅ Summary ready!")
            except Exception as e:
                st.error(f"❌ Summary error: {e}")

if st.session_state.summary:
    st.markdown("---")
    st.subheader("🎯 Literature Review Summary")
    col1, col2 = st.columns([3,1])
    
    with col1:
        with st.container(height=400):
            st.markdown("**Summary:**")
            st.markdown(st.session_state.summary.get('summary', ''))
        st.caption(f"📊 {st.session_state.summary.get('word_count', 0)} words")
    
    with col2:
        st.metric("Retrieval", f"{st.session_state.summary.get('retrieval_latency', 0):.0f}ms")
        st.metric("Generation", f"{st.session_state.summary.get('generation_latency', 0):.0f}ms")

if st.button("🔍 System Status"):
    try:
        ollama = requests.get(f"{BACKEND}/ollama-status", timeout=5).json()
        debug = requests.get(f"{BACKEND}/debug", timeout=5).json()
        st.success(f"✅ {debug.get('communities', 0)} communities + {debug.get('articles', 0)} articles\nOllama: {ollama.get('status', 'unknown')}")
    except:
        st.error("❌ Backend unavailable")
