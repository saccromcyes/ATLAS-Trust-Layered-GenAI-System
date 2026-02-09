from pathlib import Path
from pypdf import PdfReader

def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return "\n".join(parts)
    raise ValueError(f"Unsupported file type: {suffix}")

def list_files(folder: Path):
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".txt", ".md", ".pdf"]:
            yield p
