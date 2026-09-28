def chunk_text(text: str, chunk_size: int = 80, overlap: int = 15) -> list[str]:
    """Split text into word-based chunks with some shared context."""
    
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))

    return chunks


if __name__ == "__main__":
    sample = (
        "A document assistant searches your files for useful information. "
        "It splits documents into chunks, finds the chunks related to a question, "
        "and gives those passages to an AI model."
    )

    for number, chunk in enumerate(
        chunk_text(sample, chunk_size=12, overlap=3),
        start=1
    ):
        print(f"Chunk {number}: {chunk}")