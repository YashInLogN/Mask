from dataclasses import dataclass

from pathlib import Path

from document_loader import load_text_document

from text_chunks import chunk_text


@dataclass(frozen="true")
class DocumentChunk:
    source: str
    chunk_index: int
    text: str

def ingest_text_document(path: str | Path, *, chunk_size: int = 80, overlap: int = 15) -> list[DocumentChunk]:
    document = load_text_document(path)

    chunk_texts = chunk_text(
        text= document.text,
        chunk_size=chunk_size,
        overlap=overlap
    )

    return [
        DocumentChunk(
            source=document.source,
            chunk_index=index,
            text=chunk
        ) for index, chunk in enumerate(chunk_texts)
    ]

if __name__ == "__main__":
    sample_path = Path("documents") / "return_policy.txt"
    chunks = ingest_text_document(sample_path, chunk_size=30, overlap=5)

    for chunk in chunks:
        print(f"Source: {chunk.source} | Chunk: {chunk.chunk_index}")
        print(chunk.text)
        print()

