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

# Optional local embedding (sentence-transformers)
LOCAL_EMBEDDINGS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBEDDINGS_AVAILABLE = True
except ImportError:
    LOCAL_EMBEDDINGS_AVAILABLE = False

# Load environment variables
load_dotenv()

# ---------- Helper Utilities ----------
def get_openai_api_key():
    """Get OpenAI API key from Streamlit secrets or environment."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (Exception, KeyError):
        return os.getenv("OPENAI_API_KEY")

def safe_sleep(seconds):
    """Sleep with progress indication."""
    with st.spinner(f"Waiting {seconds} seconds..."):
        time.sleep(seconds)

def retry_with_backoff(func, *args, retries=5, initial_delay=1.0, backoff_factor=2.0, **kwargs):
    """
    Retry a function with exponential backoff.
    Handles rate limits and transient errors.
    """
    delay = initial_delay
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == retries - 1:
                raise
            error_msg = str(e).lower()
            if any(tok in error_msg for tok in ("rate limit", "429", "timeout", "connection")):
                st.warning(f"Attempt {attempt + 1}/{retries} failed. Retrying in {delay:.1f}s...")
                safe_sleep(delay)
                delay *= backoff_factor
            else:
                raise

# ---------- PDF Processing ----------
def extract_texts_from_pdfs(uploaded_files) -> List[str]:
    """Extract text from uploaded PDF files, handling encrypted PDFs."""
    texts = []
    for file in uploaded_files:
        try:
            reader = PdfReader(file)
            
            # Handle encrypted PDFs
            if reader.is_encrypted:
                try:
                    reader.decrypt("")  # Try empty password
                except Exception:
                    st.warning(f"Skipped encrypted PDF: {getattr(file, 'name', 'file')}")
                    continue
            
            # Extract text from each page
            pages = []
            for page in reader.pages:
                page_text = page.extract_text() or ""  # Fallback to empty string
                pages.append(page_text.strip())
            
            full_text = "\n".join(pages).strip()
            if full_text:
                texts.append(full_text)
            
            # Rewind file for later use
            file.seek(0)
            
        except Exception as e:
            st.warning(f"Could not read {getattr(file, 'name', 'file')}: {str(e)}")
    return texts

def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end == text_length:
            break
            
        start = end - overlap
        if start < 0:
            start = 0
    
    return chunks

def build_all_chunks(texts: List[str], chunk_size: int, overlap: int, max_chunks: int) -> List[Tuple[str, int, int]]:
    """Build chunks from all texts with metadata."""
    all_chunks = []
    for doc_idx, text in enumerate(texts):
        chunks = split_text_into_chunks(text, chunk_size, overlap)
        for chunk_idx, chunk in enumerate(chunks):
            if len(all_chunks) >= max_chunks:
                return all_chunks
            all_chunks.append((chunk, doc_idx, chunk_idx))
    return all_chunks

# ---------- Embeddings ----------
def openai_embed_batch(texts: List[str], model="text-embedding-3-small", batch_size=32, api_key=None) -> List[List[float]]:
    """Get embeddings from OpenAI API."""
    if not api_key:
        raise ValueError("OpenAI API key required")
    
    openai.api_key = api_key
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        def call_api():
            return openai.Embedding.create(
                model=model,
                input=batch
            )
        
        response = retry_with_backoff(call_api)
        embeddings.extend([item["embedding"] for item in response["data"]])
    
    return embeddings

def local_embed_batch(texts: List[str], model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=16):
    """Get embeddings from local model."""
    if not LOCAL_EMBEDDINGS_AVAILABLE:
        raise RuntimeError("Local embeddings require sentence-transformers")
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size)
    return embeddings.tolist()

# ---------- FAISS Index ----------
def create_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatL2:
    """Create FAISS index from embeddings."""
    if not embeddings:
        raise ValueError("No embeddings provided")
    
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index

def compute_index_key(uploaded_files, chunk_size, overlap, model_key, max_chunks) -> str:
    """Compute stable cache key from files and settings."""
    h = hashlib.sha256()
    h.update(str(time.time()).encode())  # Include timestamp
    h.update(str(chunk_size).encode())
    h.update(str(overlap).encode())
    h.update(str(max_chunks).encode())
    h.update(model_key.encode())
    
    for file in uploaded_files:
        try:
            file.seek(0)
            data = file.read()  # Read full content
            file.seek(0)  # Rewind
            h.update(getattr(file, "name", "").encode())
            h.update(str(len(data)).encode())
            # Include hash of first and last 128 bytes
            h.update(data[:128] if isinstance(data, (bytes, bytearray)) else str(data)[:128].encode())
            h.update(data[-128:] if isinstance(data, (bytes, bytearray)) else str(data)[-128:].encode())
        except Exception as e:
            st.warning(f"Could not hash file {getattr(file, 'name', 'file')}: {str(e)}")
    
    return h.hexdigest()

# ---------- Retrieval & Answering ----------
def knn_search(index: faiss.IndexFlatL2, query_emb: List[float], top_k: int = 4) -> List[int]:
    """Find top-k similar embeddings."""
    if not query_emb:
        raise ValueError("Empty query embedding")
    
    distances, indices = index.search(np.array([query_emb]).astype("float32"), top_k)
    return indices[0].tolist()

def build_prompt_with_context(question: str, retrieved_chunks: List[Tuple[str, int, int]], max_context_chars=3000) -> str:
    """Build LLM prompt with retrieved context."""
    context_parts = []
    total_length = 0
    
    for chunk, doc_idx, chunk_idx in retrieved_chunks:
        chunk_length = len(chunk)
        remaining_space = max_context_chars - total_length
        
        if remaining_space <= 0:
            break
            
        if chunk_length > remaining_space:
            chunk = chunk[:remaining_space]
            chunk_length = remaining_space
            
        context_parts.append(
            f"Source (Document {doc_idx + 1}, Chunk {chunk_idx + 1}):\n{chunk}"
        )
        total_length += chunk_length
    
    context = "\n\n---\n\n".join(context_parts)
    
    return f"""You are a helpful assistant that answers questions using only the provided context.
If the answer isn't in the context, say you don't know but suggest related information.

Context:
{context}

Question: {question}

Answer concisely in 1-3 paragraphs. Cite sources when possible."""

def call_openai_chat(prompt: str, api_key: str, model="gpt-3.5-turbo", temperature=0.1, max_tokens=1000):
    """Query OpenAI chat model."""
    if not api_key:
        raise ValueError("OpenAI API key required")
    
    openai.api_key = api_key
    messages = [
        {"role": "system", "content": "You are a helpful research assistant."},
        {"role": "user", "content": prompt}
    ]
    
    def chat_completion():
        return openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    response = retry_with_backoff(chat_completion)
    return response["choices"][0]["message"]["content"].strip()

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(
        page_title="PDF Q&A Assistant",
        page_icon="📚",
        layout="centered"
    )
    
    st.title("📚 Document Q&A Assistant")
    st.write("Upload PDFs and ask questions about their content.")
    
    # Initialize session state
    if "faiss_index_cache" not in st.session_state:
        st.session_state.faiss_index_cache = {}
    
    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        
        # API key and model selection
        openai_key = get_openai_api_key()
        st.write(f"OpenAI key: {'✅' if openai_key else '❌'}")
        
        embedding_backend = st.radio(
            "Embeddings backend",
            ["OpenAI", "Local (sentence-transformers)"],
            index=0,
            disabled=not LOCAL_EMBEDDINGS_AVAILABLE
        )
        
        if embedding_backend == "Local (sentence-transformers)" and not LOCAL_EMBEDDINGS_AVAILABLE:
            st.error("Install sentence-transformers for local embeddings")
        
        # Document processing settings
        st.subheader("Document Processing")
        chunk_size = st.slider("Chunk size (characters)", 200, 2000, 800, 50)
        chunk_overlap = st.slider("Chunk overlap (characters)", 0, 500, 200, 10)
        max_chunks = st.number_input("Maximum chunks", 100, 10000, 2000, 50)
        
        # Embedding settings
        st.subheader("Embedding Settings")
        batch_size = st.number_input(
            "Batch size",
            1,
            256,
            value=16 if embedding_backend == "Local (sentence-transformers)" else 32,
            step=1
        )
        top_k = st.number_input("Retrieve top-k chunks", 1, 10, 4, 1)
        
        # LLM settings
        st.subheader("Answer Generation")
        llm_model = st.selectbox(
            "LLM Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0
        )
        
        st.markdown("---")
        st.caption("Note: Large documents may take time to process.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyze"
    )
    
    # Early validation for OpenAI key
    if embedding_backend == "OpenAI" and not openai_key:
        st.error("🔑 OpenAI API key required. Set OPENAI_API_KEY in secrets.")
        st.stop()
    
    # Process documents when files are uploaded
    if uploaded_files and st.button("Process Documents", type="primary"):
        with st.spinner("Processing documents..."):
            try:
                # Extract text from PDFs
                texts = extract_texts_from_pdfs(uploaded_files)
                if not texts:
                    st.error("No readable text extracted from documents")
                    st.stop()
                
                # Split into chunks
                raw_chunks = build_all_chunks(texts, chunk_size, chunk_overlap, max_chunks)
                chunk_texts = [chunk for chunk, _, _ in raw_chunks]
                
                st.success(f"Processed {len(chunk_texts)} chunks from {len(texts)} documents")
                
                # Compute cache key
                model_key = f"{embedding_backend}::{embedding_backend.lower()}"
                index_key = compute_index_key(uploaded_files, chunk_size, chunk_overlap, model_key, max_chunks)
                
                # Check cache
                if index_key in st.session_state.faiss_index_cache:
                    st.info("Using cached embeddings")
                else:
                    # Create embeddings
                    with st.spinner("Creating embeddings..."):
                        progress_bar = st.progress(0)
                        
                        try:
                            if embedding_backend == "OpenAI":
                                embeddings = openai_embed_batch(
                                    chunk_texts,
                                    batch_size=batch_size,
                                    api_key=openai_key
                                )
                                emb_model = "text-embedding-3-small"
                            else:
                                embeddings = local_embed_batch(
                                    chunk_texts,
                                    batch_size=batch_size
                                )
                                emb_model = "sentence-transformers/all-MiniLM-L6-v2"
                            
                            # Build FAISS index
                            with st.spinner("Building search index..."):
                                index = create_faiss_index(embeddings)
                                
                                # Store in cache
                                st.session_state.faiss_index_cache[index_key] = {
                                    "index": index,
                                    "meta": {
                                        "chunks_meta": raw_chunks,
                                        "texts": chunk_texts,
                                        "embedding_model": emb_model
                                    }
                                }
                            
                            st.success("Index created successfully")
                            
                        except Exception as e:
                            st.error(f"Failed to create embeddings: {str(e)}")
                            raise
                        finally:
                            progress_bar.empty()
            
            except Exception as e:
                st.error(f"Document processing failed: {str(e)}")
                st.stop()
    
    # Question answering section
    if "faiss_index_cache" in st.session_state and st.session_state.faiss_index_cache:
        st.divider()
        st.subheader("Ask a Question")
        
        # Get the latest index
        latest_key = list(st.session_state.faiss_index_cache.keys())[-1]
        cache_data = st.session_state.faiss_index_cache[latest_key]
        index = cache_data["index"]
        meta = cache_data["meta"]
        
        question = st.text_input("Enter your question:")
        
        if st.button("Get Answer") and question:
            with st.spinner("Searching documents..."):
                try:
                    # Embed the question
                    if embedding_backend == "OpenAI":
                        q_embedding = openai_embed_batch(
                            [question],
                            model="text-embedding-3-small",
                            batch_size=1,
                            api_key=openai_key
                        )[0]
                    else:
                        q_embedding = local_embed_batch([question], batch_size=1)[0]
                    
                    # Retrieve relevant chunks
                    chunk_indices = knn_search(index, q_embedding, top_k)
                    retrieved_chunks = []
                    for idx in chunk_indices:
                        if idx < len(meta["chunks_meta"]):
                            retrieved_chunks.append(meta["chunks_meta"][idx])
                    
                    if not retrieved_chunks:
                        st.warning("No relevant content found")
                        st.stop()
                    
                    # Show retrieved chunks
                    with st.expander("View Retrieved Context"):
                        for chunk, doc_idx, chunk_idx in retrieved_chunks:
                            st.caption(f"Document {doc_idx + 1}, Chunk {chunk_idx + 1}")
                            st.text(chunk[:500] + ("..." if len(chunk) > 500 else ""))
                            st.divider()
                    
                    # Generate answer
                    with st.spinner("Generating answer..."):
                        prompt = build_prompt_with_context(question, retrieved_chunks)
                        answer = call_openai_chat(
                            prompt,
                            api_key=openai_key,
                            model=llm_model
                        )
                        
                        st.subheader("Answer")
                        st.write(answer)
                
                except Exception as e:
                    st.error(f"Failed to generate answer: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.stop()
                    
