import requests
from typing import List, Dict
from ..settings import settings

class OpenAIClient:
    """Minimal OpenAI-compatible client (works with OpenAI, gateways, many hosted APIs)."""

    def __init__(self):
        self.base = settings.OPENAI_BASE_URL.rstrip("/")
        self.key = settings.OPENAI_API_KEY

    def chat(self, messages: List[Dict[str, str]], model: str | None = None, temperature: float = 0.2) -> str:
        url = f"{self.base}/chat/completions"
        payload = {"model": model or settings.OPENAI_MODEL, "messages": messages, "temperature": temperature}
        r = requests.post(url, headers={"Authorization": f"Bearer {self.key}"}, json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def embed(self, texts: List[str], model: str | None = None) -> List[List[float]]:
        url = f"{self.base}/embeddings"
        payload = {"model": model or settings.OPENAI_EMBED_MODEL, "input": texts}
        r = requests.post(url, headers={"Authorization": f"Bearer {self.key}"}, json=payload, timeout=180)
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]
