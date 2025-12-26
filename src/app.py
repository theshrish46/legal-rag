import os
import streamlit as st
from datetime import datetime
from uuid import uuid4
from pathlib import Path

# Fast PDF Processing
from pypdf import PdfReader

# Environment
from dotenv import load_dotenv

# LangChain & Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    ChatHuggingFace,
    HuggingFaceEndpoint,
)
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

# --- RERANKING (The fix for "Random Results") ---
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank

# 1. SETUP PAGE & ENV
st.set_page_config(layout="wide", page_title="Legal AI Auditor")
load_dotenv()

# 2. SESSION STATE (Only for UI caching, not logic blocking)
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


# --- CACHED RESOURCES (Speed Fix) ---
@st.cache_resource
def get_qdrant_client():
    client = QdrantClient(
        url=os.getenv("CLUSTER_ENDPOINT"), api_key=os.getenv("QDRANT_API_KEY")
    )
    if not client.collection_exists(collection_name="legal-rag"):
        client.create_collection(
            collection_name="legal-rag",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
    return client


@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


@st.cache_resource
def get_llm():
    return HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-V3", task="text-generation", temperature=0.5
    )


# Initialize
qdrant_client = get_qdrant_client()
embedding_model = get_embedding_model()
llm = get_llm()

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="legal-rag",
    embedding=embedding_model,
)

# --- THE RETRIEVER SETUP ---
# 1. Fetch 15 docs (Wide Net)
base_retriever = vector_store.as_retriever(search_kwargs={"k": 15})

# 2. Re-rank top 3 (Sniper) - Fixes the "Random Things" issue
compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2")
retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# --- UI ---
st.title("Legal RAG")

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Legal Docs", accept_multiple_files=True, type="pdf"
    )

    if uploaded_files:
        for uploaded in uploaded_files:
            if uploaded.name in st.session_state.processed_files:
                continue

            # Robust Metadata
            clean_name, _ = os.path.splitext(uploaded.name)
            parts = clean_name.split("_")

            if len(parts) < 7:
                st.error(f"Format Error: {uploaded.name}")
                continue

            try:
                date_str = datetime.strptime(parts[1], "%Y%m%d").isoformat()
            except ValueError:
                date_str = datetime.now().isoformat()

            meta_data = {
                "company": parts[0],
                "date": date_str,
                "filling_type": parts[2],
                "doc_title": parts[6],
                "source_filename": uploaded.name,
            }

            with st.status(f"Processing {uploaded.name}...", expanded=True) as status:
                try:
                    pdf_reader = PdfReader(uploaded)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() or ""

                    text = " ".join(text.replace("\n", " ").split()).lower().strip()
                    chunks = splitter.split_text(text)

                    new_docs = []
                    uuids = []
                    for chunk in chunks:
                        new_docs.append(
                            Document(page_content=chunk, metadata=meta_data)
                        )
                        uuids.append(str(uuid4()))

                    # Add to Qdrant (Persistent)
                    vector_store.add_documents(documents=new_docs, ids=uuids)
                    st.session_state.processed_files.add(uploaded.name)

                    status.update(label="Indexed!", state="complete", expanded=False)

                except Exception as e:
                    st.error(f"Error: {e}")

# --- RETRIEVAL ---
st.divider()
input_text = st.text_input("Your Message")

if input_text:
    # No more "Upload First" check. If Qdrant has data, it searches.
    try:
        results = retriever.invoke(input_text)

        if not results:
            st.warning("No relevant documents found in the database.")
        else:
            st.success(f"Found {len(results)} relevant chunks.")
            for doc in results:
                # Flashrank moves metadata to 'doc.metadata' directly
                filename = doc.metadata.get("source_filename", "Unknown")
                score = doc.metadata.get(
                    "relevance_score", "N/A"
                )  # Flashrank adds this

                with st.expander(f"Source: {filename} (Score: {score})"):
                    st.write(doc.page_content)
    except Exception as e:
        st.error(f"Search Error: {e}")
