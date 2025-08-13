# main.py
"""
Streamlit PDF Q&A app (no LangChain)
- Upload PDF(s)
- Extract text, split into chunks
- Create embeddings (OpenAI or local SentenceTransformers)
- Build FAISS index and cache in session
- Answer queries using retrieved context + OpenAI chat
"""

import os
import time
import math
import hashlib
import json
from typing import List, Tuple
import streamlit as st
from PyPDF2 import PdfReader
import numpy as np
import faiss
import openai
from dotenv import load_dotenv
from tqdm import tqdm

# Optional local embedding (sentence-transformers)
LOCAL_EMBEDDINGS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBEDDINGS_AVAILABLE = True
except Exception:
    LOCAL_EMBEDDINGS_AVAILABLE = False

# Load local .env in dev (Streamlit Secrets will override in cloud)
load_dotenv()

# ---------- Helper utilities ----------
def get_openai_api_key():
    # First check Streamlit secrets, else environment
    key = None
    try:
        key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        key = os.getenv("OPENAI_API_KEY")
    return key

def safe_sleep(backoff):
    time.sleep(backoff)

def retry_with_backoff(func, *args, retries=6, initial_delay=1.0, factor=2.0, **kwargs):
    delay = initial_delay
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            # Retry on typical transient errors (rate limit, timeout, connection)
            if any(tok in msg for tok in ("rate limit", "429", "timeout", "timed out", "service unavailable", "connection")):
                if i == retries - 1:
                    raise
                safe_sleep(delay)
                delay *= factor
                continue
            else:
                raise

# ---------- PDF text extraction ----------
def extract_texts_from_pdfs(uploaded_files) -> List[str]:
    texts = []
    for f in uploaded_files:
        try:
            reader = PdfReader(f)
            pages = []
            for p in reader.pages:
                page_text = p.extract_text()
                if page_text:
                    pages.append(page_text)
            text = "\n".join(pages).strip()
            if text:
                texts.append(text)
            else:
                st.warning(f"No readable text found in {getattr(f, 'name', 'file')}. It may be scanned or image-only.")
        except Exception as e:
            st.warning(f"Failed to read {getattr(f, 'name', 'file')}: {e}")
    return texts

# ---------- Chunking ----------
def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def build_all_chunks(texts: List[str], chunk_size: int, overlap: int, max_chunks: int) -> List[Tuple[str,int,int]]:
    # returns list of (chunk_text, source_doc_index, chunk_index_in_doc)
    all_chunks = []
    for doc_idx, t in enumerate(texts):
        chunks = split_text_into_chunks(t, chunk_size, overlap)
        for i, c in enumerate(chunks):
            all_chunks.append((c, doc_idx, i))
            if len(all_chunks) >= max_chunks:
                return all_chunks
    return all_chunks

# ---------- Embeddings ----------
def openai_embed_batch(texts: List[str], model="text-embedding-3-small", batch_size=32, api_key=None) -> List[List[float]]:
    if api_key is None:
        raise ValueError("OpenAI API key required for OpenAI embeddings")
    openai.api_key = api_key
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        def call():
            return openai.Embedding.create(model=model, input=batch)
        resp = retry_with_backoff(call)
        for item in resp["data"]:
            embeddings.append(item["embedding"])
    return embeddings

def local_embed_batch(texts: List[str], model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=64):
    if not LOCAL_EMBEDDINGS_AVAILABLE:
        raise RuntimeError("Local embedding model not installed. Add sentence-transformers & torch to requirements.")
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=False, batch_size=batch_size)
    return embs.tolist()

# ---------- FAISS helpers ----------
def create_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatL2:
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    arr = np.array(embeddings).astype("float32")
    index.add(arr)
    return index

# A helper to compute a stable cache key from files + settings
def compute_index_key(uploaded_files, chunk_size, overlap, model_key, max_chunks):
    h = hashlib.sha256()
    h.update(str(chunk_size).encode())
    h.update(str(overlap).encode())
    h.update(str(max_chunks).encode())
    h.update(model_key.encode())
    for f in uploaded_files:
        # include filename + size + last bytes for small change detection
        name = getattr(f, "name", "") or ""
        try:
            f.seek(0)
            data = f.read()
            f.seek(0)
            h.update(name.encode())
            h.update(str(len(data)).encode())
            # add a slice to avoid hashing huge files fully
            h.update(data[:256] if isinstance(data, (bytes, bytearray)) else str(data)[:256].encode())
        except Exception:
            # best-effort
            h.update(name.encode())
    return h.hexdigest()

# ---------- Retrieval & Answering ----------
def knn_search(index: faiss.IndexFlatL2, query_emb: List[float], top_k: int=4) -> List[int]:
    q = np.array([query_emb]).astype("float32")
    D, I = index.search(q, top_k)
    return I[0].tolist()  # indices

def build_prompt_with_context(question: str, retrieved_chunks: List[Tuple[str,int,int]], max_context_chars=3000) -> str:
    # Compose context from retrieved chunks; limit total size
    ctx_pieces = []
    total = 0
    for chunk, doc_idx, chunk_idx in retrieved_chunks:
        add_len = len(chunk)
        if total + add_len > max_context_chars:
            # trim chunk if needed
            piece = chunk[: max(0, max_context_chars - total)]
            ctx_pieces.append(f"Source (doc {doc_idx} chunk {chunk_idx}):\n{piece}")
            total = max_context_chars
            break
        else:
            ctx_pieces.append(f"Source (doc {doc_idx} chunk {chunk_idx}):\n{chunk}")
            total += add_len
    context = "\n\n---\n\n".join(ctx_pieces)
    prompt = (
        "You are a helpful assistant that answers questions using only the provided context from the user's documents.\n"
        "If the answer is not contained in the context, say you don't know or provide best-effort with a clear note.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely and reference which Source (doc/chunk) you used when possible."
    )
    return prompt

def call_openai_chat(prompt: str, api_key: str, model="gpt-3.5-turbo", temperature=0.0, max_tokens=512):
    openai.api_key = api_key
    # messages format
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}]
    def call():
        return openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    resp = retry_with_backoff(call)
    return resp["choices"][0]["message"]["content"].strip()

# ---------- Streamlit UI & Flow ----------
st.set_page_config(page_title="PDF Q&A (OpenAI + FAISS)", layout="centered")
st.title("📄 Document Q&A Chatbot")
st.write("Upload PDFs and ask questions. This app builds a FAISS index from your docs and retrieves context for answers.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    openai_key = get_openai_api_key()
    st.write("OpenAI key:", "✅ found" if openai_key else "❌ not found")
    embedding_backend = st.radio("Embeddings backend", ("OpenAI", "Local (sentence-transformers)"))
    if embedding_backend == "Local (sentence-transformers)":
        if not LOCAL_EMBEDDINGS_AVAILABLE:
            st.error("Local embeddings not available. Install sentence-transformers & torch.")
    chunk_size = st.slider("Chunk size (chars)", 200, 2000, 1000, step=50)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 500, 200, step=10)
    max_chunks = st.number_input("Max chunks to embed", min_value=50, max_value=10000, value=2000, step=50)
    batch_size = st.number_input("Embedding batch size", min_value=1, max_value=256, value=32, step=1)
    top_k = st.number_input("Retrieval top_k", min_value=1, max_value=10, value=4, step=1)
    # LLM selection
    llm_model = st.selectbox("LLM model for answers", ("gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"), index=0)
    st.markdown("---")
    st.write("Notes:")
    st.write("- For large documents, consider Local embeddings to reduce OpenAI cost / rate-limits.")
    st.write("- On Streamlit Cloud, add OPENAI_API_KEY in Settings → Secrets.")

# File uploader
uploaded_files = st.file_uploader("📂 Upload PDF files (multiple allowed)", type=["pdf"], accept_multiple_files=True)

# Process button (explicit)
process_now = st.button("Process uploaded PDFs into index")

# Auto-process when files uploaded and not yet indexed
auto_process = False
if uploaded_files and ("last_upload_count" not in st.session_state or st.session_state.get("last_upload_count") != len(uploaded_files)):
    # mark change and auto-process
    st.session_state["last_upload_count"] = len(uploaded_files)
    auto_process = True

if process_now or auto_process:
    if not uploaded_files:
        st.warning("Upload one or more PDFs first.")
    else:
        st.info("Extracting text from PDFs...")
        texts = extract_texts_from_pdfs(uploaded_files)
        if not texts:
            st.error("No readable text extracted from uploaded PDFs.")
            st.stop()

        st.info("Splitting into chunks...")
        raw_chunks = build_all_chunks(texts, chunk_size, chunk_overlap, int(max_chunks))
        chunk_texts = [c for (c, _, _) in raw_chunks]
        st.write(f"Built {len(chunk_texts)} chunks from {len(texts)} documents.")

        # compute index cache key
        model_key = embedding_backend + ("::openai" if embedding_backend=="OpenAI" else "::local")
        index_key = compute_index_key(uploaded_files, chunk_size, chunk_overlap, model_key, max_chunks)

        # see if index in session cache
        if "faiss_index_cache" not in st.session_state:
            st.session_state.faiss_index_cache = {}

        if index_key in st.session_state.faiss_index_cache:
            st.success("Found cached index — using it.")
        else:
            st.info("Creating embeddings and building FAISS index...")
            progress = st.progress(0)
            try:
                if embedding_backend == "OpenAI":
                    if not openai_key:
                        st.error("OpenAI API key missing. Set OPENAI_API_KEY in env or Streamlit Secrets.")
                        st.stop()
                    # openai embedding model
                    emb_model = "text-embedding-3-small"
                    embeddings = openai_embed_batch(chunk_texts, model=emb_model, batch_size=int(batch_size), api_key=openai_key)
                else:
                    if not LOCAL_EMBEDDINGS_AVAILABLE:
                        st.error("Local embeddings not available on this system.")
                        st.stop()
                    embeddings = local_embed_batch(chunk_texts, batch_size=int(batch_size))

                # build faiss index
                st.info("Indexing embeddings into FAISS...")
                index = create_faiss_index(embeddings)
                # store metadata: texts list, and mapping indices -> (text, doc_idx, chunk_idx)
                meta = {
                    "chunks_meta": raw_chunks,  # list of (text, doc_idx, chunk_idx)
                    "texts": chunk_texts
                }
                st.session_state.faiss_index_cache[index_key] = {"index": index, "meta": meta, "embedding_model": emb_model if embedding_backend=="OpenAI" else "local"}
                st.success("Index built and cached.")
            except Exception as e:
                st.exception(e)
            finally:
                progress.empty()

# If any index in cache, let user query
if "faiss_index_cache" in st.session_state and st.session_state.faiss_index_cache:
    # choose the most recent index (just pick the last key)
    last_key = list(st.session_state.faiss_index_cache.keys())[-1]
    cached = st.session_state.faiss_index_cache[last_key]
    index = cached["index"]
    meta = cached["meta"]
    st.write("Index ready. You can now ask questions.")
    question = st.text_input("💬 Ask a question about the uploaded documents:")
    if st.button("Get Answer") and question:
        # embed the question
        try:
            if embedding_backend == "OpenAI":
                if not openai_key:
                    st.error("OpenAI key missing.")
                    st.stop()
                q_emb = openai_embed_batch([question], model=cached.get("embedding_model","text-embedding-3-small"), batch_size=1, api_key=openai_key)[0]
            else:
                if not LOCAL_EMBEDDINGS_AVAILABLE:
                    st.error("Local embeddings not available.")
                    st.stop()
                q_emb = local_embed_batch([question], batch_size=1)[0]

            # search
            idxs = knn_search(index, q_emb, top_k=int(top_k))
            retrieved = []
            for i in idxs:
                if i < len(meta["chunks_meta"]):
                    retrieved.append(meta["chunks_meta"][i])
            # build prompt
            prompt = build_prompt_with_context(question, retrieved)
            st.write("## Retrieved context:")
            for (c, didx, cidx) in retrieved:
                st.write(f"- Doc {didx} chunk {cidx} — {len(c)} chars")
            # call LLM
            if not openai_key:
                st.error("OpenAI key required to generate answer. Add it in Streamlit Secrets or environment.")
                st.stop()
            answer = call_openai_chat(prompt, api_key=openai_key, model=llm_model)
            st.write("### Answer")
            st.write(answer)
        except Exception as e:
            st.exception(e)
else:
    st.info("No index yet. Upload PDFs and click 'Process uploaded PDFs into index' (or upload to auto-process).")

# small footer
st.markdown("---")
st.caption("This app builds a local FAISS index for retrieval and uses OpenAI chat to answer. Use Local embeddings to avoid embedding API costs.")
    
