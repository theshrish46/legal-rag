# System imports
import os
from pathlib import Path
import time
from datetime import datetime

import streamlit as st

# Handling Uploaded Files
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

# env
from dotenv import load_dotenv

load_dotenv()

# Text Chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LLM Models
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    ChatHuggingFace,
    HuggingFaceEndpoint,
)

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

qdrant_client = QdrantClient(
    url=os.getenv("CLUSTER_ENDPOINT"), api_key=os.getenv("QDRANT_API_KEY")
)

if not qdrant_client.collection_exists(collection_name="legal-rag"):
    print(f"Collection exists legal-rag doesn't. Creating it now...")
    qdrant_client.create_collection(
        collection_name="legal-rag",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )
else:
    print(f"Collection legal-rag already exists.")

print(qdrant_client.get_collections())


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

embedding_model = HuggingFaceEmbeddings(model="google/embeddinggemma-300m")

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.2", task="text-generation")
chat_model = ChatHuggingFace(llm=llm)

vectore_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="legal-rag",
    embedding=embedding_model,
    force_recreate=True,
)


DATA_DIR = Path("DATA/UPLOADED")
DATA_DIR.mkdir(parents=True, exist_ok=True)

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "generated_pdf" not in st.session_state:
    st.session_state.generated_pdf = None

st.set_page_config(layout="wide", page_title="Legal AI Auditor")
st.title("Legal RAG")


with st.sidebar:
    st.title("Legal RAG")
    st.header("Upload your legal documents here")
    uploaded_files = st.file_uploader(
        "Legal Doc's", accept_multiple_files=True, type="pdf"
    )

for uploaded in uploaded_files:
    if uploaded is not None:
        st.write(uploaded.name.split("_"))

        parts = uploaded.name.split("_")

        if len(parts) < 7:
            st.error("Error fetching meta data")
        date = parts[1]
        st.write(date)
        date_str = datetime.strptime(date, "%Y%m%d").isoformat()

        meta_data = {
            "company": parts[0],
            "date": date_str,
            "filling_type": parts[2],
            "doc_title": parts[6],
            "source_filename": uploaded.name,
        }

        file_path = DATA_DIR / uploaded.name
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded.getbuffer())
        except Exception as e:
            st.write(e)

        with st.status(
            "Extracting Data...", width="stretch", expanded=True, state="running"
        ) as status:
            text = ""
            images = convert_from_path(str(file_path))
            for img in images:
                text += pytesseract.image_to_string(img)
            status.update(label="Extracted Data", state="complete", expanded=False)
            st.success("Text Extracted Successfully!!")

        text = text.replace("\n", " ")
        text = " ".join(text.split())
        cleaned_text = text.lower().strip()
        cleaned_text = splitter.split_text(cleaned_text)
        st.write(len(cleaned_text))
        st.write(type(cleaned_text))
