from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from local_embeddings import load_embedding_model


COLLECTION_NAME = "document_chunks"
PROJECT_DIRECTORY = Path(__file__).resolve().parent
QDRANT_PATH = PROJECT_DIRECTORY / "storage" / "qdrant"


def open_vector_store(vector_size: int) -> QdrantClient:
    """Open the local vector database and ensure its collection is ready."""
    if vector_size <= 0:
        raise ValueError("vector_size must be greater than zero")

    client = QdrantClient(path=str(QDRANT_PATH))

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
    else:
        collection = client.get_collection(COLLECTION_NAME)
        vector_config = collection.config.params.vectors

        if (
            vector_config.size != vector_size
            or vector_config.distance != Distance.COSINE
        ):
            client.close()
            raise ValueError(
                "The existing collection's vector configuration "
                "does not match this embedding model"
            )

    return client


if __name__ == "__main__":
    model = load_embedding_model()
    client = open_vector_store(vector_size=model.embedding_size)

    print(f"Local database path: {QDRANT_PATH}")
    print(f"Collection ready: {COLLECTION_NAME}")
    print(f"Vector dimensions: {model.embedding_size}")

    client.close()