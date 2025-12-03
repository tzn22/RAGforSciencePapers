# frontend/app_streamlit.py
import streamlit as st
import os, requests
from dotenv import load_dotenv
load_dotenv()

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Local Sci RAG", layout="wide")
st.title("Local RAG — no API keys, local LLM")

q = st.text_input("Query", "")
k = st.slider("Top K", min_value=1, max_value=10, value=5)

col1, col2 = st.columns([3,1])
with col1:
    if st.button("Search") and q.strip():
        resp = requests.post(f"{BACKEND}/query", json={"q": q, "top_k": k}, timeout=120).json()
        st.write("latency:", resp.get("latency"))
        for i, r in enumerate(resp.get("results", []), 1):
            st.markdown(f"**{i}. {r['title']}**  (score: {r['score']:.4f})")
            st.caption(r.get("abstract","")[:400])
            if st.button(f"View {i}"):
                st.write(r.get("text",""))
        if st.button("Summarize (local LLM)"):
            s = requests.post(f"{BACKEND}/summarize", json={"q": q, "top_k": k}, timeout=300).json()
            st.markdown("### Summary")
            st.write(s.get("summary",""))
with col2:
    st.info("Instructions")
    st.markdown("""
    1. Run ingestion: `python ingestion/prepare_corpus.py --sample 5000` (sample for dev).\n
    2. Start backend: `uvicorn backend.app.main:app --reload`.\n
    3. Start this app: `streamlit run frontend/app_streamlit.py`.\n
    Note: local LLM (Qwen2.5) may require GPU for acceptable speed.
    """)
