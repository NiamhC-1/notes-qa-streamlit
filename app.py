import re
import json
import pickle
from pathlib import Path
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Notes Q&A Agent", page_icon="🔎", layout="wide")

DOCS_DIR = Path("docs")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def chunk(text: str, max_chars: int = 900):
    parts, cur = [], ""
    for seg in re.split(r"(?<=[.!?])\s+|\n{2,}", text):
        seg = seg.strip()
        if not seg:
            continue
        if len(cur) + len(seg) + 1 <= max_chars:
            cur += (" " if cur else "") + seg
        else:
            parts.append(cur)
            cur = seg
    if cur:
        parts.append(cur)
    return parts

def read_docs():
    items = []
    for p in DOCS_DIR.glob("**/*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                txt = p.read_text(errors="ignore")
            items.append((str(p), txt))
    return items

def file_signature():
    sig = []
    for p in DOCS_DIR.glob("**/*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            stat = p.stat()
            sig.append((str(p), int(stat.st_mtime), stat.st_size))
    sig.sort()
    return tuple(sig)

@st.cache_resource(show_spinner=False)
def load_model(name: str):
    return SentenceTransformer(name)

@st.cache_data(show_spinner=True)
def build_index(sig: tuple, model_name: str, max_chars: int = 900, n_neighbors: int = 5):
    raw = read_docs()
    paths, chunks = [], []
    for path, txt in raw:
        for i, c in enumerate(chunk(txt, max_chars=max_chars)):
            paths.append(path)
            chunks.append(c)
    if not chunks:
        return {"paths": [], "chunks": [], "emb": None, "nn": None, "count_files": 0, "count_chunks": 0}
    model = load_model(model_name)
    emb = model.encode(chunks, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(chunks)), metric="cosine").fit(emb)
    return {
        "paths": paths,
        "chunks": chunks,
        "emb": emb,
        "nn": nn,
        "count_files": len(set(paths)),
        "count_chunks": len(chunks),
    }

def retrieve(query: str, k: int, model_name: str, index):
    if not index["chunks"]:
        return 0.0, []
    model = load_model(model_name)
    qv = model.encode([query], normalize_embeddings=True)
    dists, idxs = index["nn"].kneighbors(qv, n_neighbors=min(k, len(index["chunks"])))
    sims = 1 - dists[0]
    hits = [(float(sims[i]), (index["paths"][j], index["chunks"][j])) for i, j in enumerate(idxs[0])]
    conf = float(sims.max()) if len(sims) else 0.0
    return conf, hits

def summarize_from_hits(query: str, hits, k_sent: int = 4):
    q = set(re.findall(r"\w+", query.lower()))
    cands = []
    for score, (path, text) in hits:
        for s in re.split(r"(?<=[.!?])\s+", text):
            toks = set(re.findall(r"\w+", s.lower()))
            overlap = len(q & toks)
            if overlap:
                cands.append((overlap, s.strip(), path))
    if not cands:
        for _, (path, text) in hits[:2]:
            for s in re.split(r"(?<=[.!?])\s+", text)[:2]:
                cands.append((1, s.strip(), path))
    cands.sort(key=lambda x: x[0], reverse=True)
    picked = cands[:k_sent]
    body = " ".join(s for _, s, _ in picked)
    sources = []
    seen = set()
    for _, _, p in picked:
        if p not in seen:
            sources.append(p)
            seen.add(p)
    return body, sources

st.title("🔎 Notes Q&A Agent (Semantic)")
st.caption("Ask natural-language questions over your local .txt/.md notes.")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Top passages (k)", min_value=1, max_value=10, value=3)
    max_chars = st.slider("Chunk size (characters)", min_value=400, max_value=1500, value=900, step=50)
    threshold = st.slider("Low-confidence threshold", 0.0, 1.0, 0.30, 0.01)
    st.markdown("---")
    if st.button("Rebuild index"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.experimental_rerun()
    st.markdown("**Docs folder:** `docs/`")
    st.markdown("Add `.txt` / `.md` files to the repo's `docs` folder and click *Rebuild index*.")

sig = file_signature()
index = build_index(sig, MODEL_NAME, max_chars=max_chars)

if index['count_chunks']:
    st.success(f"Indexed {index['count_files']} files, {index['count_chunks']} chunks.")
else:
    st.warning("No .txt/.md files found in `docs/`.")

query = st.text_input("Ask a question about your notes")
ask = st.button("Ask")

if ask and query:
    conf, hits = retrieve(query, k, MODEL_NAME, index)
    if conf < threshold:
        st.info(f"Not sure (confidence {conf:.2f}). Try rephrasing or add more notes.")
    if hits:
        answer, sources = summarize_from_hits(query, hits)
        st.subheader("Answer")
        st.write(answer if answer else "No answer could be generated from the top passages.")
        st.subheader("Sources")
        for s in sources:
            with st.expander(s):
                snippet = ""
                for sc, (p, text) in hits:
                    if p == s:
                        snippet = text[:800] + (" …" if len(text) > 800 else "")
                        break
                st.write(snippet)
    else:
        st.warning("No relevant passages found.")
