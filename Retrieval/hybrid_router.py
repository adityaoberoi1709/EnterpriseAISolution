import hashlib
import json
import logging
from typing import List, Literal

import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings
from config.LLM_Factory import get_llm
from Ingestion.vector_store import get_vector_store
from Retrieval.graph_retrieval import get_graph_retrieval, EntitySearchInput

logger = logging.getLogger(__name__)

QueryIntent = Literal["entity", "semantic", "hybrid"]

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Classify this retrieval query into exactly one of:
    -"entity" : asks about specific named enities, people , organizations, relationships
    -"semantic" : asks for concepts, explanations, summaries, how/why questions
    -"hybrid" : complex queries needing both entity lookup and semantic understanding
    
    Reply with ONLY the single word : entity, semantic, or hybrid."""),
    ("human", "{query}")
])


def classify_intent(query: str) -> QueryIntent:
    lower = query.lower()
    entity_signals = ["who is", "what is", "relationship between", "connected to", "related to"]
    semantic_signals = ["how does", "explain", "summarize", "what are the steps", "why"]
    if any(s in lower for s in entity_signals):
        return "entity"
    if any(s in lower for s in semantic_signals):
        return "semantic"
    try:
        llm = get_llm()
        chain = INTENT_PROMPT | llm
        response = chain.invoke({"query": query})
        intent = response.content.strip().lower()
        if intent not in ("entity", "semantic", "hybrid"):
            intent = "hybrid"
        logger.info(f"Query Intent classified as {intent}")
        return intent
    except Exception as e:
        logger.warning(f"Intent classification failed: {e}, defaulting to hybrid")
        return "hybrid"


def rerank(query: str, docs: List[Document], top_n: int | None = None) -> List[Document]:
    top_n = top_n or settings.RERANKER_TOP_N
    if len(docs) <= top_n:
        return docs
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, doc.page_content) for doc in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        reranked = [doc for _, doc in ranked[:top_n]]
        return reranked
    except ImportError:
        logger.warning("sentence-transformers not available, using score-based dedup")
        # Deduplicate by content hash and return top_n
        seen = set()
        unique = []
        for doc in docs:
            h = hashlib.md5(doc.page_content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(doc)
        return unique[:top_n]


def reciprocal_rank_fusion(vector_docs: List[Document], graph_docs: List[Document], alpha: float | None = None, k: int = 60) -> List[Document]:
    alpha = alpha or settings.HYBRID_ALPHA
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}
    for rank, doc in enumerate(vector_docs):
        key = hashlib.md5(doc.page_content.encode()).hexdigest()
        scores[key] = scores.get(key, 0) + alpha * (1 / (k + rank + 1))
        doc_map[key] = doc
    
    for rank, doc in enumerate(graph_docs):
        key = hashlib.md5(doc.page_content.encode()).hexdigest()
        scores[key] = scores.get(key, 0) + (1 - alpha) * (1 / (k + rank + 1))
        doc_map[key] = doc
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    fused = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = scores[key]
        fused.append(doc)
    return fused


class HybridRouter:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.graph_retriever = get_graph_retrieval()

    def retrieve(self, query: str, mode: QueryIntent = None, top_k: int = None, use_cache: bool = False) -> List[Document]:
        top_k = top_k or settings.RERANKER_TOP_N
        mode = mode or classify_intent(query)
        vector_docs: List[Document] = []
        graph_docs: List[Document] = []
        if mode in ("semantic", "hybrid"):
            try:
                vector_docs = self.vector_store.similarity_search(query, k=settings.FAISS_TOP_K)
                for d in vector_docs:
                    d.metadata["retrieval_source"] = "vector"
            except Exception as e:
                logger.warning(f"Vector Retrieval Failed : {e}")
        if mode in ("entity", "hybrid"):
            try:
                entity_inp = EntitySearchInput(entity_name=query.split()[0], limit=10)
                graph_docs = self.graph_retriever.entity_search(entity_inp)
                if not graph_docs:
                    graph_docs = self.graph_retriever.fulltext_search(query, limit=10)
            except Exception as e:
                logger.warning(f"Graph retrieval failed : {e}")
        if mode == "hybrid" and vector_docs and graph_docs:
            fused = reciprocal_rank_fusion(vector_docs, graph_docs)
        elif mode == "entity":
            fused = graph_docs or vector_docs
        else:
            fused = vector_docs or graph_docs
        final = rerank(query, fused, top_n=top_k)
        return final


_router_instance: HybridRouter | None = None


def get_router() -> HybridRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = HybridRouter()
    return _router_instance
