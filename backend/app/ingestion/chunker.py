from typing import List, Dict
import re
import uuid

def chunk_text(text: str, chunk_size: int = 950, overlap: int = 140) -> List[Dict]:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[Dict] = []
    buf = ""

    for p in paras:
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append({"chunk_id": f"chk_{uuid.uuid4().hex[:12]}", "text": buf})
            if len(p) > chunk_size:
                start = 0
                while start < len(p):
                    end = min(len(p), start + chunk_size)
                    chunks.append({"chunk_id": f"chk_{uuid.uuid4().hex[:12]}", "text": p[start:end]})
                    start = max(end - overlap, end)
                buf = ""
            else:
                buf = p

    if buf:
        chunks.append({"chunk_id": f"chk_{uuid.uuid4().hex[:12]}", "text": buf})
    return chunks
