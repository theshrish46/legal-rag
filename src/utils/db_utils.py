from qdrant_client import QdrantClient, models


def is_file_indexed(client: QdrantClient, collection_name: str, filename: str) -> bool:
    """Checks if segments of this file already exist in Qdrant."""
    try:
        # Check for the filename in the metadata field
        search_result = client.count(
            collection_name=collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.source_filename",
                        match=models.MatchValue(value=filename),
                    )
                ]
            ),
        )
        return search_result.count > 0
    except Exception:
        return False
