import os
from datetime import datetime
from uuid import uuid4
from pypdf import PdfReader
from langchain_core.documents import Document

def extract_legal_metadata(file_name: str):
    """Parses legal metadata from structured filenames."""
    clean_name, _ = os.path.splitext(file_name)
    parts = clean_name.split("_")
    
    if len(parts) < 7:
        return None

    try:
        date_str = datetime.strptime(parts[1], "%Y%m%d").isoformat()
    except ValueError:
        date_str = datetime.now().isoformat()

    return {
        "company": parts[0],
        "date": date_str,
        "filling_type": parts[2],
        "doc_title": parts[6],
        "source_filename": file_name,
    }

def process_pdf_to_documents(uploaded_file, splitter):
    """Extracts text from PDF, cleans it, and returns a list of Document objects."""
    # 1. Extract
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # 2. Clean (Senior trick: normalize whitespace once)
    text = " ".join(text.replace("\n", " ").split()).lower().strip()
    
    # 3. Chunk
    chunks = splitter.split_text(text)
    
    # 4. Metadata
    meta = extract_legal_metadata(uploaded_file.name)
    if not meta:
        raise ValueError(f"Invalid filename format: {uploaded_file.name}")

    # 5. Contextualize Chunks
    new_docs = []
    for chunk in chunks:
        # We prepend metadata to the chunk text so the embedding 'knows' the context
        contextualized_text = (
            f"Doc: {meta['doc_title']} | "
            f"Company: {meta['company']} | "
            f"Content: {chunk}"
        )
        new_docs.append(
            Document(page_content=contextualized_text, metadata=meta)
        )
    
    return new_docs