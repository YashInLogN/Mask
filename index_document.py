from collections.abc import Sequence
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from ingestion import DocumentChunk, ingest_text_document
from local_embeddings import embed_texts, load_embedding_model
from vector_store import COLLECTION_NAME, open_vector_store


def store_chunks(
    client: QdrantClient,
    chunks: Sequence[DocumentChunk],
    embeddings: np.ndarray,
) -> int:
    """Store chunk vectors and their source metadata in Qdrant."""
    matrix = np.asarray(embeddings, dtype=np.float32)

    if not chunks:
        raise ValueError("chunks must contain at least one item")
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional matrix")
    if matrix.shape[0] != len(chunks):
        raise ValueError("there must be exactly one vector for each chunk")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings must contain only finite numbers")

    collection = client.get_collection(COLLECTION_NAME)
    vector_config = collection.config.params.vectors

    if matrix.shape[1] != vector_config.size:
        raise ValueError(
            f"Expected vectors with {vector_config.size} values, "
            f"got {matrix.shape[1]}"
        )

    points = []

    for index, chunk in enumerate(chunks):
        stable_key = f"{chunk.source}:{chunk.chunk_index}"
        point_id = str(uuid5(NAMESPACE_URL, stable_key))

        points.append(
            PointStruct(
                id=point_id,
                vector=matrix[index].tolist(),
                payload={
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    return len(points)


if __name__ == "__main__":
    document_path = Path("documents") / "return_policy.txt"
    chunks = ingest_text_document(
        document_path,
        chunk_size=30,
        overlap=5,
    )

    model = load_embedding_model()
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(model, chunk_texts)

    client = open_vector_store(vector_size=model.embedding_size)

    try:
        stored_count = store_chunks(client, chunks, embeddings)
    finally:
        client.close()

    print(f"Stored {stored_count} chunks from {document_path}")