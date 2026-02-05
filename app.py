import streamlit as st
from src.database import get_vector_store
from src.retriever import get_legal_retriever
from src.prompts.legal_templates import get_rag_chain
from src.ingestion import process_pdf_to_documents
from src.text_handler.splitter import get_recursive_text_splitter
from src.utils.db_utils import is_file_indexed
from src.utils.chat_utils import convert_to_langchain_messages
from uuid import uuid4

# --- 1. PAGE CONFIG & ENGINE CACHING ---
st.set_page_config(layout="wide", page_title="Legal AI Auditor", page_icon="⚖️")


@st.cache_resource
def load_engine():
    """Initializes the heavy components only once."""
    vector_store = get_vector_store()
    retriever = get_legal_retriever()
    chain = get_rag_chain()
    splitter = get_recursive_text_splitter(chunk_size=2000, chunk_overlap=400)
    return vector_store, retriever, chain, splitter


vector_store, retriever, chain, splitter = load_engine()

# --- 2. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. SIDEBAR: DOCUMENT INGESTION ---
with st.sidebar:
    st.title("📂 Document Center")
    uploaded_files = st.file_uploader(
        "Upload Legal PDFs", accept_multiple_files=True, type="pdf"
    )

    if uploaded_files:
        for file in uploaded_files:
            # Prevent Duplicates
            if is_file_indexed(vector_store.client, "legal-rag", file.name):
                st.info(f"✔ {file.name} is already indexed.")
                continue

            with st.status(f"Analyzing {file.name}...", expanded=False) as status:
                try:
                    docs = process_pdf_to_documents(file, splitter)
                    ids = [str(uuid4()) for _ in range(len(docs))]
                    vector_store.add_documents(documents=docs, ids=ids)
                    status.update(label=f"✅ {file.name} indexed!", state="complete")
                    count = vector_store.client.count(collection_name="legal-rag").count
                    st.write(f"Confirmed points in cloud: {count}")
                except Exception as e:
                    st.error(f"Failed to index {file.name}: {e}")

# --- 4. CHAT INTERFACE ---
st.title("⚖️ Legal RAG Auditor")
st.caption("PhD-level reasoning for contract auditing and legal discovery.")

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask a legal question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Response Generation
    with st.chat_message("assistant"):
        # We pass the question and converted chat history to the chain
        # The chain handles the retrieval and formatting internally!
        history = convert_to_langchain_messages(st.session_state.messages[:-1])

        response_stream = chain.stream({"question": prompt, "chat_history": history})

        full_response = st.write_stream(response_stream)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
