from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from ..settings import settings

def get_qdrant() -> QdrantClient:
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

def ensure_collection(client: QdrantClient, dim: int):
    names = [c.name for c in client.get_collections().collections]
    if settings.QDRANT_COLLECTION not in names:
        client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

def upsert_chunks(vectors: List[List[float]], payloads: List[Dict]):
    if not vectors:
        return
    client = get_qdrant()
    ensure_collection(client, dim=len(vectors[0]))
    points = [PointStruct(id=pl["chunk_id"], vector=vec, payload=pl) for vec, pl in zip(vectors, payloads)]
    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
