import os
import streamlit as st
from datetime import datetime
from uuid import uuid4
from pathlib import Path


# Fast PDF Processing
from pypdf import PdfReader
from langchain_core.documents import Document


# DB Imports
from qdrantDB.qdrant_db import get_vector_store
from qdrantDB.retriever import get_contextual_compression_retriever
from qdrantDB.utils import is_file_indexed
from prompts.legal_prompt import format_docs, get_rag_chain

# Text Handler Import
from text_handler.text_splitter import get_recursive_text_splitter


# --- RERANKING (The fix for "Random Results") ---


# 1. SETUP PAGE & ENV
st.set_page_config(layout="wide", page_title="Legal AI Auditor")

# 2. SESSION STATE (Only for UI caching, not logic blocking)
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


if "messages" not in st.session_state:
    st.session_state.messages = []

# --- THE RETRIEVER SETUP ---
# 1. Fetch 15 docs (Wide Net)
vector_store = get_vector_store()

# 2. Re-rank top 3 (Sniper) - Fixes the "Random Things" issue
retriever = get_contextual_compression_retriever()

splitter = get_recursive_text_splitter(chunk_size=500, chunk_overlap=100)

# --- UI ---
st.title("Legal RAG")

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Legal Docs", accept_multiple_files=True, type="pdf"
    )

    if uploaded_files:
        for uploaded in uploaded_files:
            if is_file_indexed(vector_store.client, "legal-rag", uploaded.name):
                st.toast(f"Skipping {uploaded.name} already exsists")
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
                        contextualized_text = (
                            f"Document Title : {meta_data['doc_title']} | "
                            f"Date : {meta_data['date']} |"
                            f"Company : {meta_data['company']} |"
                            f"{chunk}"
                        )
                        new_docs.append(
                            Document(
                                page_content=contextualized_text, metadata=meta_data
                            )
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


# 1. Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Handle Input
if prompt := st.chat_input("Ask about your legal docs...", key="first_chat"):
    # Add User Message to State
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show User Message
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. RAG Logic
    results = retriever.invoke(prompt)

    if results:
        context_text = format_docs(results)

        # Ensure you pass your 'chat_model' here if your function requires it
        # e.g., rag_chain = get_rag_chain(chat_model)
        rag_chain = get_rag_chain()

        with st.chat_message("assistant"):
            # Create the stream generator
            response_stream = rag_chain.stream(
                {"context": context_text, "question": prompt, "chat_history": []}
            )

            # --- THE FIX ---
            # st.write_stream writes to UI AND returns the final string
            full_response = st.write_stream(response_stream)

        # Append the STRING 'full_response', not the stream object
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
    else:
        # Handle case where no docs are found
        with st.chat_message("assistant"):
            st.warning("No relevant documents found.")
