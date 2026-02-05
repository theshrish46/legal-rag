import os
from langchain_huggingface import HuggingFaceEmbeddings

# Global variable to hold the model in memory
_embedding_model_instance = None


def get_embedding_model():
    global _embedding_model_instance

    if _embedding_model_instance is None:
        # Determine device (use GPU if available, else CPU)
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {"device": device}
        encode_kwargs = {
            "normalize_embeddings": True
        }  # Essential for Cosine Similarity

        _embedding_model_instance = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        print(f"Embedding model loaded on {device}")

    return _embedding_model_instance
