from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from src.database import get_vector_store


def get_legal_retriever():
    """
    Two-stage retrieval:
    1. Similarity Search (Fetch 25 candidates)
    2. FlashRank Reranking (Refine to top 5)
    """
    # Initialize the base vector store
    vector_store = get_vector_store()

    # Stage 1: The "Wide Net" (Retrieves 25 docs based on vector similarity)
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 25})

    # Stage 2: The "Sniper" (Reranker)
    # Using TinyBERT-L2-v2 as it's lightning fast for legal text
    compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2", top_n=5)

    # Combine them
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    return compression_retriever
