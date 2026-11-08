# Single-container image: runs on Hugging Face Spaces, Render, Fly.io, Cloud Run or a laptop.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE_DIR=/home/user/app/.cache/models
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

COPY --chown=user atlas ./atlas
COPY --chown=user corpus ./corpus
COPY --chown=user web ./web
COPY --chown=user eval ./eval

# Download the embedding and reranking models and pre-build the demo index at build time,
# so a cold start serves the first question in seconds instead of downloading ~200 MB.
RUN python -c "from atlas.config import get_settings; from atlas.main import Services; s = Services(get_settings()); assert type(s.embedder).__name__ == 'FastEmbedEmbedder' and s.reranker.name == 'cross-encoder', 'models failed to load'; s.bootstrap_corpus()"

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/health')"
CMD ["uvicorn", "atlas.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
