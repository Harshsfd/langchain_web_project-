# main.py — Streamlit PDF Q&A with rate-limit safe embeddings + local fallback
import os
import time
import hashlib
import streamlit as st
from PyPDF2 import PdfReader

# --- LangChain imports with compatibility for old/new versions ---
try:
    # Newer split
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
except Exception:
    from langchain.embeddings.openai import OpenAIEmbeddings  # old
    from langchain.chat_models import ChatOpenAI  # old

try:
    from langchain_community.vectorstores import FAISS  # new location
except Exception:
    from langchain.vectorstores import FAISS  # old location

try:
    # Splitter
    from langchain.text_splitter import RecursiveCharacterTextSplitter as TextSplitter
except Exception:
    from langchain.text_splitter import CharacterTextSplitter as TextSplitter

# Local CPU embeddings (no API calls)
HF_EMBEDDING_AVAILABLE = True
try:
    try:
        # new location
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except Exception:
        from langchain.embeddings import HuggingFaceEmbeddings  # old
except Exception:
    HF_EMBEDDING_AVAILABLE = False

# ------------- Config -------------
st.set_page_config(page_title="📄 PDF Q&A Chatbot", layout="centered")
st.title("📄 Document Q&A Chatbot")
st.caption("LangChain + OpenAI • Handles rate limits • Optional local embeddings")

# Get API key (Cloud: Secrets, Local: env)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# UI controls
with st.expander("⚙️ Settings"):
    embedding_backend = st.radio(
        "Embedding backend",
        ["OpenAI (recommended)", "Local (no API limits)"],
        index=0 if OPENAI_API_KEY else 1,
        help="Use Local if you hit OpenAI rate limits. Local runs fully on CPU."
    )
    chunk_size = st.slider("Chunk size (characters)", 400, 2000, 900, 50)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, 100, 10)
    max_chunks = st.number_input("Max chunks to embed (to guard quotas)", 50, 5000, 1200, 50)
    batch_size = st.number_input("Embedding batch size", 1, 128, 32, 1)
    use_small_embed = st.checkbox("Use smaller OpenAI embedding model", True,
                                  help="text-embedding-3-small (cheaper & lighter)")

uploaded_files = st.file_uploader(
    "📂 Upload PDF files", type=["pdf"], accept_multiple_files=True
)

# -------- Helpers --------
def parse_pdfs(files):
    texts = []
    for f in files:
        try:
            reader = PdfReader(f)
            content = []
            for p in reader.pages:
                content.append(p.extract_text() or "")
            text = "\n".join(content).strip()
            if text:
                texts.append(text)
            else:
                st.warning(f"⚠️ No readable text found in **{f.name}** (maybe scanned without OCR).")
        except Exception as e:
            st.warning(f"⚠️ Could not read **{getattr(f, 'name', 'file')}**: {e}")
    return texts

def split_texts(texts, size, overlap):
    splitter = TextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def hash_chunks(chunks, backend_key):
    # Build a stable key so we can cache FAISS in session_state
    h = hashlib.sha256()
    h.update(backend_key.encode())
    for c in chunks[:2000]:  # limit for speed
        h.update(str(len(c)).encode()); h.update(c[:100].encode('utf-8', errors='ignore'))
    return h.hexdigest()

def build_openai_embedder():
    # Force a specific model to reduce rate-limit pressure
    model_name = "text-embedding-3-small" if use_small_embed else "text-embedding-3-large"
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=model_name)

def build_local_embedder():
    if not HF_EMBEDDING_AVAILABLE:
        raise RuntimeError("HuggingFaceEmbeddings not installed. Add 'sentence-transformers' to requirements.txt")
    # Light, fast CPU model
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def faiss_from_texts_batched(texts, embedder, batch=32, status_cb=None):
    """
    Build a FAISS index incrementally to handle rate limits gracefully.
    Retries with exponential backoff on 429 errors.
    """
    vs = None
    for start in range(0, len(texts), batch):
        sub = texts[start:start+batch]
        # retry loop per batch
        delay = 1.5
        for attempt in range(6):  # up to ~1 + 3 + 6 + 12 + 24 + 48 ≈ 94s worst-case
            try:
                if vs is None:
                    vs = FAISS.from_texts(sub, embedder)
                else:
                    vs.add_texts(sub)
                if status_cb:
                    status_cb(min(start + batch, len(texts)), len(texts))
                break
            except Exception as e:
                msg = str(e)
                # heuristic: handle rate-limits & transient network
                if "RateLimit" in msg or "429" in msg or "Temporary" in msg or "timeout" in msg.lower():
                    time.sleep(delay)
                    delay *= 2
                    continue
                # unrecoverable
                raise
    return vs

def make_llm():
    # Keep compatibility with different LangChain versions:
    # Some use "model_name", newer use "model".
    model_choice = st.selectbox("LLM model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    try:
        return ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name=model_choice)
    except TypeError:
        return ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model=model_choice)

def make_retriever(vs):
    return vs.as_retriever(search_kwargs={"k": 4})

def build_chain(vs):
    # Import where it exists
    try:
        from langchain.chains import RetrievalQA
    except Exception:
        from langchain.chains.retrieval_qa.base import RetrievalQA
    llm = make_llm()
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=make_retriever(vs))

# -------- Main flow --------
if uploaded_files:
    texts = parse_pdfs(uploaded_files)
    if not texts:
        st.stop()

    chunks = split_texts(texts, chunk_size, chunk_overlap)
    if len(chunks) > max_chunks:
        st.info(f"✂️ Trimming from {len(chunks)} to {max_chunks} chunks to respect quotas.")
        chunks = chunks[:max_chunks]

    if embedding_backend.startswith("OpenAI"):
        if not OPENAI_API_KEY:
            st.error("No OPENAI_API_KEY found. Switch to Local embeddings or set the key in Secrets.")
            st.stop()
        embedder = build_openai_embedder()
        backend_key = f"openai::{getattr(embedder, 'model', 'unknown')}"
    else:
        embedder = build_local_embedder()
        backend_key = "local::all-MiniLM-L6-v2"

    cache_key = f"faiss::{hash_chunks(chunks, backend_key)}::{batch_size}"
    if "faiss_store" not in st.session_state:
        st.session_state.faiss_store = {}
    vs = st.session_state.faiss_store.get(cache_key)

    if vs is None:
        st.write("🔧 Building embeddings index…")
        progress = st.progress(0)
        info = st.empty()

        def _status(done, total):
            progress.progress(int((done / total) * 100))
            info.write(f"Embedding {done}/{total} chunks…")

        try:
            vs = faiss_from_texts_batched(chunks, embedder, batch=int(batch_size), status_cb=_status)
        except Exception as e:
            # Helpful guidance on rate-limit or missing deps
            if "RateLimit" in str(e) or "429" in str(e):
                st.error("Hit OpenAI rate limits while embedding. Try:\n"
                         "• Lower batch size\n"
                         "• Smaller chunk size\n"
                         "• Switch to **Local (no API limits)** embeddings in Settings")
                st.stop()
            elif "HuggingFaceEmbeddings" in str(e):
                st.error("Local embeddings not available. Add `sentence-transformers` to requirements.txt.")
                st.stop()
            else:
                st.exception(e)
                st.stop()

        progress.empty(); info.empty()
        st.session_state.faiss_store[cache_key] = vs

    st.success("✅ Index ready. Ask your question:")
    query = st.text_input("💬 Your question")
    if query:
        qa_chain = build_chain(vs)
        with st.spinner("🤔 Thinking…"):
            try:
                answer = qa_chain.run(query)
            except Exception as e:
                if "RateLimit" in str(e) or "429" in str(e):
                    st.error("LLM rate-limited while answering. Wait a few seconds and try again.")
                    st.stop()
                raise
        st.markdown(f"**Answer:** {answer}")
else:
    st.info("⬆️ Upload one or more PDFs to start.")
        
