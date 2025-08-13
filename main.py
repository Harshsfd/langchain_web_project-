# main.py — Streamlit PDF Q&A with robust retriever wrapper (fixes AttributeError)
import os
import time
import hashlib
import streamlit as st
from PyPDF2 import PdfReader

# --- LangChain imports with compatibility for old/new versions ---
# Embeddings & Chat LLM
try:
    # newer split packaging
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
except Exception:
    # older packaging
    try:
        from langchain.embeddings.openai import OpenAIEmbeddings
    except Exception:
        OpenAIEmbeddings = None
    try:
        from langchain.chat_models import ChatOpenAI
    except Exception:
        ChatOpenAI = None

# Vectorstore: try community and fallback
try:
    from langchain_community.vectorstores import FAISS as FAISSCommunity  # community package
except Exception:
    FAISSCommunity = None

try:
    from langchain.vectorstores import FAISS as FAISSCore
except Exception:
    FAISSCore = None

FAISS = FAISSCommunity or FAISSCore

# Text splitter compatibility
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter as TextSplitter
except Exception:
    from langchain.text_splitter import CharacterTextSplitter as TextSplitter

# Local CPU embeddings (HuggingFace)
HF_EMBEDDING_AVAILABLE = True
try:
    try:
        # community/new location
        from langchain_community.embeddings import HuggingFaceEmbeddings as HFEmbCommunity
    except Exception:
        from langchain.embeddings import HuggingFaceEmbeddings as HFEmbCore
        HFEmbCommunity = None
    HuggingFaceEmbeddings = HFEmbCommunity or HFEmbCore
except Exception:
    HuggingFaceEmbeddings = None
    HF_EMBEDDING_AVAILABLE = False

# RetrievalQA import fallback
try:
    from langchain.chains import RetrievalQA
except Exception:
    try:
        from langchain.chains.retrieval_qa.base import RetrievalQA
    except Exception:
        RetrievalQA = None

# ------------- Config -------------
st.set_page_config(page_title="📄 PDF Q&A Chatbot", layout="centered")
st.title("📄 Document Q&A Chatbot")
st.caption("LangChain + OpenAI • Robust across versions • Local embeddings optional")

# Get API key (Cloud: Secrets, Local: env)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# --- UI controls ---
with st.expander("⚙️ Settings"):
    embedding_backend = st.radio(
        "Embedding backend",
        ["OpenAI (recommended)", "Local (no API limits)"],
        index=0 if OPENAI_API_KEY else 1,
        help="If you hit OpenAI rate limits, switch to Local."
    )
    chunk_size = st.slider("Chunk size (characters)", 400, 2000, 900, 50)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, 100, 10)
    max_chunks = st.number_input("Max chunks to embed (guard quotas)", 50, 5000, 1200, 50)
    batch_size = st.number_input("Embedding batch size", 1, 128, 32, 1)
    use_small_embed = st.checkbox("Use smaller OpenAI embedding model", True,
                                  help="text-embedding-3-small is cheaper and faster")

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
                st.warning(f"⚠️ No readable text in **{getattr(f, 'name', 'file')}** (might be scanned image).")
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
    h = hashlib.sha256()
    h.update(backend_key.encode())
    for c in chunks[:2000]:
        h.update(str(len(c)).encode())
        h.update(c[:100].encode('utf-8', errors='ignore'))
    return h.hexdigest()


def build_openai_embedder():
    if OpenAIEmbeddings is None:
        raise RuntimeError("OpenAIEmbeddings not available in this environment. Check installed packages.")
    model_name = "text-embedding-3-small" if use_small_embed else "text-embedding-3-large"
    # try both arg names: some versions accept model, others model_name
    try:
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=model_name)
    except TypeError:
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model_name=model_name)


def build_local_embedder():
    if not HF_EMBEDDING_AVAILABLE or HuggingFaceEmbeddings is None:
        raise RuntimeError("Local HuggingFace embeddings not available. Add 'sentence-transformers' to requirements.")
    # Use a small, fast embedding model for CPU
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except TypeError:
        return HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")


def faiss_from_texts_batched(texts, embedder, batch=32, status_cb=None):
    """
    Build FAISS index in batches. Retries on transient errors (e.g., RateLimit).
    """
    if FAISS is None:
        raise RuntimeError("FAISS vectorstore class not found. Check requirements (faiss-cpu + langchain).")

    vs = None
    for start in range(0, len(texts), batch):
        sub = texts[start:start + batch]
        delay = 1.5
        for attempt in range(6):
            try:
                if vs is None:
                    # initial build
                    vs = FAISS.from_texts(sub, embedder)
                else:
                    # many FAISS impls have add_texts; if not, fallback to creating a new index
                    if hasattr(vs, "add_texts"):
                        vs.add_texts(sub)
                    else:
                        # rebuild by merging texts (simple fallback — slower)
                        all_texts = []  # gather existing texts if available
                        try:
                            if hasattr(vs, "texts"):
                                all_texts = vs.texts + sub
                            else:
                                # No stored texts: we can't merge; create new index from previously embedded data not available
                                # In practice, add_texts should exist — but if not, re-create from scratch:
                                # NOTE: this fallback rebuilds from provided slices only (less ideal)
                                vs = FAISS.from_texts(sub, embedder)
                        except Exception:
                            vs = FAISS.from_texts(sub, embedder)
                if status_cb:
                    status_cb(min(start + batch, len(texts)), len(texts))
                break
            except Exception as e:
                msg = str(e)
                if "RateLimit" in msg or "429" in msg or "Temporary" in msg or "timeout" in msg.lower():
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
    return vs


# Robust retriever wrapper — returns an object with get_relevant_documents(query)
class SimpleRetrieverWrapper:
    def __init__(self, vs, k=4):
        self.vs = vs
        self.k = k

    def get_relevant_documents(self, query):
        # Many FAISS objects expose similarity_search or similarity_search_with_score
        if hasattr(self.vs, "similarity_search"):
            try:
                return self.vs.similarity_search(query, k=self.k)
            except TypeError:
                # some impls expect (query, k) vs (query, k)
                return self.vs.similarity_search(query, k=self.k)
        elif hasattr(self.vs, "similarity_search_with_score"):
            docs_with_scores = self.vs.similarity_search_with_score(query, k=self.k)
            return [d for d, s in docs_with_scores]
        else:
            raise RuntimeError("Vector store has no similarity search method.")


def make_retriever(vs, k=4):
    # Prefer built-in as_retriever if present
    if hasattr(vs, "as_retriever"):
        try:
            return vs.as_retriever(search_kwargs={"k": k})
        except Exception:
            # some as_retriever signatures differ; try calling without args
            try:
                return vs.as_retriever()
            except Exception:
                return SimpleRetrieverWrapper(vs, k=k)
    else:
        return SimpleRetrieverWrapper(vs, k=k)


def make_llm():
    # pick model (be mindful of access on your account)
    model_choice = st.selectbox("LLM model", ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"], index=0)
    if ChatOpenAI is None:
        raise RuntimeError("ChatOpenAI not available in this environment. Check installed packages.")
    # try both creation signatures
    try:
        return ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name=model_choice)
    except TypeError:
        try:
            return ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model=model_choice)
        except TypeError:
            # last resort: omit api key param (if stored globally)
            return ChatOpenAI(temperature=0)


def build_chain(vs):
    retriever = make_retriever(vs, k=4)
    # RetrievalQA expects a retriever with get_relevant_documents
    if RetrievalQA is None:
        raise RuntimeError("Runtime error occurred")

    
