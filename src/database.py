import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from src.llm_model.embeddings import get_embedding_model

load_dotenv()

_client_instance = None
_vector_store_instance = None


def get_qdrant_client():
    """Returns a single, persistent Qdrant client instance."""
    global _client_instance

    if _client_instance is None:
        _client_instance = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        # Ensure collection exists
        collection_name = "legal-rag"
        if not _client_instance.collection_exists(collection_name):
            _client_instance.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

    return _client_instance


def get_vector_store():
    """Returns a single, persistent LangChain VectorStore instance."""
    global _vector_store_instance

    if _vector_store_instance is None:
        client = get_qdrant_client()
        embedding_model = get_embedding_model()

        _vector_store_instance = QdrantVectorStore(
            client=client,
            collection_name="legal-rag",
            embedding=embedding_model,
        )

    return _vector_store_instance
