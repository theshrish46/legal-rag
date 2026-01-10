from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank


from qdrantDB.qdrant_db import get_vector_store


def get_flash_rerank(model="ms-marco-TinyBERT-L-2-v2"):
    return FlashrankRerank(model=model, top_n=5)


compressor = get_flash_rerank()
base_retriever = get_vector_store().as_retriever(kwargs=25)


def get_contextual_compression_retriever(
    base_compressor=compressor, base_retriever=base_retriever
):
    return ContextualCompressionRetriever(
        base_compressor=base_compressor, base_retriever=base_retriever
    )
