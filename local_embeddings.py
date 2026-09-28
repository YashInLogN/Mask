from collections.abc import Sequence

import numpy as np

from fastembed import TextEmbedding

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
def load_embedding_model() -> TextEmbedding:
    return TextEmbedding(model_name=MODEL_NAME)

def embed_texts(model: TextEmbedding, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
    if isinstance(texts, str) or not isinstance(texts, Sequence):
        raise ValueError("texts must be a sequence of strings, such as a list")
    if len(texts) <= 0:
        raise ValueError("texts must contain at least one item")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if not all(isinstance(text, str) for text in texts):
        raise ValueError("every item in texts must be a string")

    cleaned_text = [text.strip() for text in texts]

    if any(not text for text in cleaned_text):
        raise ValueError("texts cannot contain empty or whitespace-only items")

    vectors = list(model.embed(cleaned_text, batch_size=batch_size))

    matrix = np.asarray(vectors, dtype=np.float32)

    expected_shape = (len(cleaned_text), model.embedding_size)

    if expected_shape != matrix.shape:
        raise RuntimeError(f"Expected embedding matrix shape {expected_shape}, got {matrix.shape}")

    if not np.isfinite(matrix).all():
        raise RuntimeError("the embedding model returned a non-finite value")

    return matrix


if __name__ == "__main__":
    sample_texts = [
        "How do I get a refund for a damaged item?",
        "Damaged products can be returned for a refund within 30 days.",
        "Our stores are open from 9 a.m. to 6 p.m.",
    ]

    model = load_embedding_model()
    embeddings = embed_texts(model, sample_texts)

    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"First five values for the question: {embeddings[0, :5].tolist()}")
    
    