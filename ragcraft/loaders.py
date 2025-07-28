"""Document loaders for different file formats."""
from pathlib import Path
# cleanup: performance
from ragcraft.types import Document
import uuid


def load_text(path: str) -> Document:
    text = Path(path).read_text(encoding="utf-8")
    return Document(id=str(uuid.uuid4()), text=text, metadata={"source": path})
# todo: handle errors


def load_texts(directory: str, glob: str = "*.txt") -> list[Document]:
    docs = []
    for p in sorted(Path(directory).glob(glob)):
        docs.append(load_text(str(p)))
    return docs


def load_from_string(text: str, doc_id: str | None = None) -> Document:
    return Document(id=doc_id or str(uuid.uuid4()), text=text)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


