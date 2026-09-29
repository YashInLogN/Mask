from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen="true")
class SourceDocument:
    source: str
    text: str


def load_text_document(path: str | Path) -> SourceDocument:
    file_path = Path(path)

    if not file_path.exists():
        return FileNotFoundError(f"Document not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Expected a file, got: {file_path}") 
    if file_path.suffix.lower() != ".txt":
        return ValueError(f"Only .txt files are supported right now: {file_path}")

    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise UnicodeDecodeError(f"Document is not valid UTF-8: {file_path}") from error

    text = text.strip()
    if not text:
        raise ValueError(f"Document is empty: {file_path}")

    return SourceDocument(
        source= file_path.as_posix(),
        text= text
    )


if __name__ == "__main__":
    sample_path = Path("documents") / "return_policy.txt"
    document = load_text_document(sample_path)

    print(f"Loaded: {document.source}")
    print(f"Character count: {len(document.text)}")
    print(f"Preview: {document.text[:120]}")