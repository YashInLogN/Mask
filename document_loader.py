from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen="true")
class SourceDocument:
    source: str
    text: str


def load_text_document(path: str | Path) -> SourceDocument:
    file_path = Path(path)

    if not file_path.exists():
        return FileNotFoundError("")
    if not file_path.is_file():
        raise ValueError("") 
    if file_path.suffix.lower() != ".txt":
        return ValueError("")

    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise UnicodeDecodeError("") from error

    text = text.strip()
    if not text:
        raise ValueError("")

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