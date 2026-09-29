from collections.abc import Sequence

import numpy as np

from local_embeddings import embed_texts, load_embedding_model

def rank_chunks(query_vector: np.ndarray, chunks: Sequence[str], chunk_vectors: np.ndarray, top_k: int = 3) -> list[tuple[int, str, float]]:
    query = np.asarray(query_vector, dtype=np.float32)
    vectors = np.asarray(chunk_vectors, dtype=np.float32)

    if isinstance(chunks, str) or not isinstance(chunks, Sequence):
        raise TypeError("chunks must be a sequence of strings")
    if not chunks:
        raise ValueError("chunks must contain at least one item")
    if not all(isinstance(chunk, str) and chunk.strip() for chunk in chunks):
        raise ValueError("every chunk must be a non-empty string")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if query.ndim != 1:
        raise ValueError("query_vector must be one-dimensional")
    if vectors.ndim != 2:
        raise ValueError("chunk_vectors must be a two-dimensional matrix")
    if vectors.shape[0] != len(chunks):
        raise ValueError("there must be exactly one vector for each chunk")
    if vectors.shape[1] != query.shape[0]:
        raise ValueError("query and chunk vectors must have the same dimensions")
    if not np.isfinite(query).all() or not np.isfinite(vectors).all():
        raise ValueError("vectors must contain only finite numbers")

    query_norm = np.linalg.norm(query)
    chunk_norms = np.linalg.norm(vectors, axis=1)

    if query_norm == 0 or np.any(chunk_norms == 0):
        raise ValueError("cosine similarity is undefined for zero vectors")

    scores = (vectors @ query) / (chunk_norms * query_norm)
    best_indices = np.argsort(-scores, kind="stable")[:top_k]

    ranked_chunks = []
    for index in best_indices:
        chunk_index = int(index)
        ranked_chunks.append(
            (chunk_index, chunks[chunk_index], float(scores[chunk_index]))
        )

    return ranked_chunks




if __name__ == "__main__":
    question = "How do I get a refund for a damaged item?"

    candidate_chunks = [
        "Damaged products can be returned for a refund within 30 days.",
        "Our stores are open from 9 a.m. to 6 p.m.",
    ]

    model = load_embedding_model()
    all_texts = [question, *candidate_chunks]
    all_vectors = embed_texts(model, all_texts)

    query_vector = all_vectors[0]
    chunk_vectors = all_vectors[1:]

    results = rank_chunks(query_vector, candidate_chunks, chunk_vectors)

    for rank, (chunk_index, chunk, score) in enumerate(results, start=1):
        print(f"Rank {rank} | score {score:.3f} | chunk {chunk_index}: {chunk}")

