import numpy as np

def cosine_similarity(query_vector: list[float], chunk_vector: list[float]) -> float:
    
    left_vector = np.asarray(query_vector, dtype=np.float64)
    right_vector = np.asarray(chunk_vector, dtype=np.float64)

    if left_vector.ndim != 1 or right_vector.ndim != 1:
        raise ValueError("Both vectors must be one- dimensional")
    if left_vector.shape != right_vector.shape:
        raise ValueError("Both vectors must have same shape to map positon by position")
    if not np.isfinite(left_vector).all() or not np.isfinite(right_vector).all:
        raise ValueError("Both vectors must contain finite numbers")

    left_norm = np.linalg.norm(left_vector)
    right_norm = np.linalg.norm(right_vector)

    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector")

    return float(np.dot(left_vector, right_vector) / (left_norm * right_norm))


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