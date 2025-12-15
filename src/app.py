import streamlit as st
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
from pathlib import Path
import time


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

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
        st.write(uploaded.name)

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
        st.write(cleaned_text)
