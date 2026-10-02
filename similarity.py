import numpy as np

def cosine_similarity(query: list[float], chunk: list[float]) -> float:
    
    query_vector = np.asarray(query, dtype=np.float64)
    chunk_vector = np.asarray(chunk, dtype=np.float64)

    if query_vector.ndim != 1 or chunk_vector.ndim != 1:
        raise ValueError("Both vectors must be one- dimensional")
    if query_vector.shape != chunk_vector.shape:
        raise ValueError("Both vectors must have same shape to map positon by position")
    if not np.isfinite(query_vector).all() or not np.isfinite(chunk_vector).all:
        raise ValueError("Both vectors must contain finite numbers")

    query_norm = np.linalg.norm(query_vector)
    chunk_norm = np.linalg.norm(chunk_vector)

    if query_norm == 0 or chunk_norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector")

    return float(np.dot(query_vector, chunk_vector) / (query_norm * chunk_norm))


if __name__ == "__main__":
    question_vector = [1.0, 0.0]

    candidates = {
        "same direction": [1.0, 0.0],
        "partly aligned": [0.8, 0.6],
        "perpendicular": [0.0, 1.0],
    }

    for label, candidate_vector in candidates.items():
        score = cosine_similarity(question_vector, candidate_vector)
        print(f"{label}: {score:.2f}")