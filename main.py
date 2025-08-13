# main.py
import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load OpenAI key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Functions ---
def load_docs(folder_path="docs"):
    docs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            pdf = PdfReader(os.path.join(folder_path, file_name))
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            docs.append(text)
    return docs

def split_docs(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for doc in docs:
        chunks.extend(text_splitter.split_text(doc))
    return chunks

def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def setup_qa_chain(vectorstore):
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa

# --- Streamlit UI ---
st.title("📄 Document Q&A Chatbot")

if "qa_chain" not in st.session_state:
    with st.spinner("Loading documents and building chatbot..."):
        docs = load_docs()
        chunks = split_docs(docs)
        vectorstore = create_vectorstore(chunks)
        st.session_state.qa_chain = setup_qa_chain(vectorstore)

query = st.text_input("Ask a question about your documents:")

if st.button("Get Answer") and query:
    with st.spinner("Thinking..."):
        answer = st.session_state.qa_chain.run(query)
    st.markdown(f"**Answer:** {answer}")
  
