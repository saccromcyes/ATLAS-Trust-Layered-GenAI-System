"""Load documents and split them into heading-aware, overlapping chunks with stable IDs."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str
    session_id: str = "demo"
    pages: list[tuple[int, int]] = field(default_factory=list)  # (char_offset, page_number) for PDFs

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    heading: str
    position: int
    session_id: str
    page: int | None = None


def slugify(name: str) -> str:
    stem = Path(name).stem.lower()
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-") or "document"


def _title_from_markdown(text: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else fallback


def load_bytes(filename: str, data: bytes, session_id: str = "demo") -> Document:
    """Turn an uploaded or on-disk file into a Document."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type '{suffix}'. Use PDF, Markdown or plain text.")

    doc_id = slugify(filename)
    fallback_title = Path(filename).stem.replace("-", " ").replace("_", " ").title()

    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        parts, pages, offset = [], [], 0
        for number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            pages.append((offset, number))
            parts.append(page_text)
            offset += len(page_text) + 2
        text = "\n\n".join(parts)
        if not text.strip():
            raise ValueError("No extractable text found in this PDF (it may be a scanned image).")
        title = (reader.metadata.title if reader.metadata and reader.metadata.title else None) or fallback_title
        return Document(doc_id, title, text, filename, session_id, pages)

    text = data.decode("utf-8", errors="replace")
    if not text.strip():
        raise ValueError("The file is empty.")
    return Document(doc_id, _title_from_markdown(text, fallback_title), text, filename, session_id)


def load_directory(path: str | Path) -> list[Document]:
    docs = []
    for file in sorted(Path(path).iterdir()):
        if file.suffix.lower() in SUPPORTED_SUFFIXES and file.is_file():
            docs.append(load_bytes(file.name, file.read_bytes()))
    return docs


_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", flags=re.MULTILINE)


def _sections(text: str) -> list[tuple[str, int, str]]:
    """Split markdown into (heading, start_offset, body) sections. Plain text is one section."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", 0, text)]
    sections = []
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        sections.append(("", 0, text[: matches[0].start()]))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((match.group(2).strip(), match.start(), text[match.start() : end]))
    return sections


def _split_long(body: str, size: int, overlap: int) -> list[tuple[int, str]]:
    """Split text into windows of ~size chars, preferring paragraph and sentence boundaries."""
    if len(body) <= size:
        return [(0, body)]
    windows, start = [], 0
    while start < len(body):
        end = min(start + size, len(body))
        if end < len(body):
            window = body[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > size * 0.5:
                end = start + cut + 1
        windows.append((start, body[start:end]))
        if end >= len(body):
            break
        start = max(end - overlap, start + 1)
    return windows


def _page_for(offset: int, pages: list[tuple[int, int]]) -> int | None:
    page = None
    for page_offset, number in pages:
        if page_offset <= offset:
            page = number
        else:
            break
    return page


def chunk_document(doc: Document, size: int = 700, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, section_start, body in _sections(doc.text):
        for local_start, piece in _split_long(body, size, overlap):
            text = piece.strip()
            if len(text) < 20:  # skip fragments such as a lone heading
                continue
            offset = section_start + local_start
            digest = hashlib.sha1(f"{doc.session_id}:{doc.doc_id}:{offset}:{text}".encode()).hexdigest()[:12]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{digest}",
                    doc_id=doc.doc_id,
                    doc_title=doc.title,
                    text=text,
                    heading=heading,
                    position=len(chunks),
                    session_id=doc.session_id,
                    page=_page_for(offset, doc.pages),
                )
            )
    return chunks
