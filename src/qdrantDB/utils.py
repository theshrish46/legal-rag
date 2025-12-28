from qdrant_client import QdrantClient
from qdrant_client.http import models


def is_file_indexed(client: QdrantClient, collection_name: str, filename: str) -> bool:
    """
    Checks if a file with the given filename already exists in the Qdrant collection.
    """
    try:
        count_result = client.count(
            collection_name=collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source_filename",  # Key path in Qdrant JSON
                        match=models.MatchValue(value=filename),
                    )
                ]
            ),
        )
        return count_result.count > 0
    except Exception as e:
        # If collection doesn't exist yet, it returns False safely
        print(f"Check failed (collection might not exist): {e}")
        return False
