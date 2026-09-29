from collections.abc import Sequence

import numpy as np

from local_embeddings import embed_texts, load_embedding_model

def rank_chunks(query_vector: np.ndarray, texts: Sequence[str], chunk_vectors: np.ndarray, top_k: int = 3) -> list[tuple[int, str, float]]:
    query = np.asarray(query_vector, dtype=np.float32)
    vectors = np.asarray(chunk_vectors, dtype= np.float32)

    if isinstance(texts, str) or not isinstance(texts, Sequence) or not all(isinstance(text, str) for text in texts):
        pass
    if query.ndim != 1 or vectors.ndim != 2:
        pass
    if vectors.shape[0] != len(texts) or vectors.shape[1] != query.shape[0]:
        pass
    if top_k <= 0:
        pass

    query_norm = np.linalg.norm(query)
    chunk_vectors = np.linalg.norm(vectors, axis= 1)

    if query_norm == 0 or np.any(chunk_vectors == 0):
        pass

    scores = (vectors @ query)/ (query_norm * chunk_vectors)
    best_indices = np.argsort(-scores, kind="stable")[:top_k]

    resulted_chunk = []

    for best in best_indices:
        index = int(best)
        resulted_chunk.append((index, texts[index], float(scores[index])))

    return resulted_chunk




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

