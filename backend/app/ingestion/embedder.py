from typing import List
from ..llm.client import OpenAIClient

class Embedder:
    def __init__(self):
        self.client = OpenAIClient()

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        bs = 64
        for i in range(0, len(texts), bs):
            out.extend(self.client.embed(texts[i:i+bs]))
        return out
