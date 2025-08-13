import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Get OpenAI API key
# First check Streamlit Secrets (for deployment), then environment variables (for local dev)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not OPENAI_API_KEY:
    st.error("❌ No OpenAI API key found. Set it in Streamlit Secrets or .env.")
    st.stop()

# --- Functions ---
def load_docs(folder_path="docs"):
    """Load all PDFs from a folder."""
    docs = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create if not exists
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            try:
                pdf = PdfReader(pdf_path)
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                if text.strip():
                    docs.append(text)
            except Exception as e:
                st.warning(f"⚠️ Could not read {file_name}: {e}")
    return docs

def split_docs(docs):
    """Split text into smaller chunks for embeddings."""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for doc in docs:
        chunks.extend(text_splitter.split_text(doc))
    return chunks

def create_vectorstore(chunks):
    """Create FAISS vector store from chunks."""
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def setup_qa_chain(vectorstore):
    """Create a RetrievalQA chain."""
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa

# --- Streamlit UI ---
st.set_page_config(page_title="📄 PDF Q&A Chatbot", layout="centered")
st.title("📄 Document Q&A Chatbot")
st.markdown("Ask questions about your uploaded PDFs using **LangChain + OpenAI**.")

# File uploader
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# Process uploaded PDFs
if uploaded_files:
    with st.spinner("Processing documents..."):
        docs = []
        for uploaded_file in uploaded_files:
            try:
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                if text.strip():
                    docs.append(text)
            except Exception as e:
                st.warning(f"⚠️ Could not process {uploaded_file.name}: {e}")

    if docs:
        chunks = split_docs(docs)
        vectorstore = create_vectorstore(chunks)
        qa_chain = setup_qa_chain(vectorstore)

        st.success("✅ Documents processed. You can now ask questions!")

        query = st.text_input("Ask a question about your documents:")

        if query:
            with st.spinner("Thinking..."):
                answer = qa_chain.run(query)
            st.markdown(f"**Answer:** {answer}")
    else:
        st.warning("⚠️ No readable text found in uploaded PDFs.")

else:
    st.info("📂 Upload one or more PDF files to get started.")
    
