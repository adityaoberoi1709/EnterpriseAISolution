import logging 
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS as LangchainFAISS

from config.settings import settings
from config.LLM_Factory import get_embeddings

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    def __init__(self, index_path: str | None = None):
        self.index_path = Path(index_path or settings.FAISS_INDEX_PATH)
        self.store: LangchainFAISS | None = None
        self.embeddings = get_embeddings()

    def build_from_documents(self, docs: List[Document]) -> None:
        self.store = LangchainFAISS.from_documents(docs, self.embeddings)
        logger.info(f"FAISS index built from {len(docs)} documents")

    def build_from_embeddings(self, docs: List[Document], embeddings: np.ndarray) -> None:
        text_embedding_pairs = [(doc.page_content, emb.tolist()) for doc, emb in zip(docs, embeddings)]
        self.store = LangchainFAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=self.embeddings,
            metadatas=[doc.metadata for doc in docs]
        )
        logger.info("FAISS index built from pre-computed embeddings")

    def add_documents(self, docs: List[Document]) -> None:
        if self.store is None:
            self.build_from_documents(docs)
            return
        self.store.add_documents(docs)
        self.save()

    #--Persistence-------
    def save(self) -> None:
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(self.index_path))

    def load(self) -> bool:
        index_file = self.index_path / "index.faiss"
        if not index_file.exists():
            logger.warning(f"No FAISS Index Found at {self.index_path}")
            return False
        self.store = LangchainFAISS.load_local(
            str(self.index_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        return True

    def similarity_search(self, query: str, k: int | None = None, filter=None) -> List[Document]:
        if self.store is None:
            raise RuntimeError("Vector Store not initalised - call build or load first")
        k = k or settings.FAISS_TOP_K
        return self.store.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(self, query: str, k: int | None = None) -> List[tuple[Document, float]]:
        if self.store is None:
            raise RuntimeError("Vector store not there")
        k = k or settings.FAISS_TOP_K
        return self.store.similarity_search_with_score(query, k=k)

    def mmr_search(self, query: str, k: int = None, fetch_k: int = 20, lambda_mult: float = 0.5) -> List[Document]:
        if self.store is None:
            raise RuntimeError("Vector store not initialized")
        k = k or settings.FAISS_TOP_K
        return self.store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)

    def as_retriever(self, search_type: str = "similarity", **kwargs):
        if self.store is None:
            raise RuntimeError("Vector store not initialised")
        return self.store.as_retriever(search_type=search_type, search_kwargs=kwargs)

    @property
    def total_vectors(self) -> int:
        if self.store is None:
            return 0
        return self.store.index.ntotal


_store_instance: FAISSVectorStore | None = None


def get_vector_store() -> FAISSVectorStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = FAISSVectorStore()
        _store_instance.load()
    return _store_instance
