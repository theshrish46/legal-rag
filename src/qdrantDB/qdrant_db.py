import os

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams


from models.models import get_embedding_model


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


def get_vector_store():
    client = get_qdrant_client()
    embedding_model = get_embedding_model()
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="legal-rag",
        embedding=embedding_model,
    )
    return vector_store
