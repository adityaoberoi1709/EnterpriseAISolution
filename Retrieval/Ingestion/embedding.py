import hashlib 
import json 
import logging 
from typing import List, Tuple

import numpy as np
from langchain_core.documents import Document

from config.settings import settings
from config.LLM_Factory import get_embeddings

logger = logging.getLogger(__name__)

    
def _get_redis():
    try:
        import redis
        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        client.ping()
        return client
    except Exception:
        logger.warning("Redis unavailable - embedding cache disabled")
        return None


def _cache_key(text: str) -> str:
    return "emb:" + hashlib.sha256(text.encode()).hexdigest()


def _get_cached(client, text: str) -> List[float] | None:
    if client is None:
        return None
    val = client.get(_cache_key(text))
    if val:
        return json.loads(val)
    return None


def _set_cached(client, text: str, embedding: List[float]) -> None:
    if client is None:
        return
    client.setex(_cache_key(text), settings.REDIS_CACHE_TTL, json.dumps(embedding))


def embed_texts(texts: List[str]) -> np.ndarray:
    embedder = get_embeddings()
    redis_client = _get_redis()
    results: List[List[float] | None] = [None] * len(texts)
    uncached_indices: List[int] = []
    uncached_texts: List[str] = []
    for i, text in enumerate(texts):
        cached = _get_cached(redis_client, text)
        if cached:
            results[i] = cached
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)
    batch_size = settings.EMBEDDING_BATCH_SIZE
    for batch_start in range(0, len(uncached_texts), batch_size):
        batch = uncached_texts[batch_start:batch_start + batch_size]
        batch_embeddings = embedder.embed_documents(batch)
        for j, (orig_idx, emb) in enumerate(zip(uncached_indices[batch_start:batch_start + batch_size], batch_embeddings)):
            results[orig_idx] = emb
            _set_cached(redis_client, uncached_texts[batch_start + j], emb)
    return np.array(results, dtype=np.float32)


def embed_docs(docs: List[Document]) -> Tuple[List[Document], np.ndarray]:
    texts = [doc.page_content for doc in docs]
    embeddings = embed_texts(texts)
    logger.info(f"Embedded {len(docs)} documents -> shape {embeddings.shape}")
    return docs, embeddings


def embed_query(query: str) -> np.ndarray:
    redis_client = _get_redis()
    cached = _get_cached(redis_client, query)
    if cached:
        return np.array(cached, dtype=np.float32)
    embedder = get_embeddings()
    vec = embedder.embed_query(query)
    _set_cached(redis_client, query, vec)
    return np.array(vec, dtype=np.float32)
