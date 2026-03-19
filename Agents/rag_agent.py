import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config.LLM_Factory import get_llm
from config.settings import settings
from Retrieval.hybrid_router import get_router

logger = logging.getLogger(__name__)


class RAGResponse(BaseModel):
    answer: str = Field(..., description="Comprehensive answer grounded in the retrieved context")
    citations: List[str] = Field(
        default_factory=list,
        description="List of source references used to generate the answer"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1 based on context relevance")
    has_sufficient_context: bool = Field(..., description="Whether retrieved context was sufficient to answer the query")


RAG_SYSTEM = """You are an enterprise knowledge assistant with access to internal documents.
Answer questions strictly based on the provided context.
Rules:
- If the context is insufficient, say so clearly and set has_sufficient_context=false
- Never fabricate information not present in the context
- Cite sources using the format [Source: <source_name>]
- Assign confidence based on how well the context supports your answer
- Be concise but complete"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM),
    ("human", """Context documents: {context}
     Question: {question}
 
    Provide a grounded answer with citations."""),
])


def format_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        parts.append(f"[{i}] Source: {source} | Chunk: {chunk_idx}\n{doc.page_content}")
    return "\n\n".join(parts)


def retrieve_context(query: str, top_k: int | None = None) -> List[Document]:
    router = get_router()
    return router.retrieve(query, top_k=top_k)


def answer(
        query: str,
        top_k: int | None = None,
        retrieval_mode=None
) -> RAGResponse:
    router = get_router()
    docs = router.retrieve(query, mode=retrieval_mode, top_k=top_k)
    if not docs:
        return RAGResponse(
            answer="I could not find relevant information in the knowledge base",
            citations=[],
            confidence=0.0,
            has_sufficient_context=False
        )
    context = format_context(docs)
    llm = get_llm()
    structured_llm = llm.with_structured_output(RAGResponse)
    chain = RAG_PROMPT | structured_llm
    try:
        response: RAGResponse = chain.invoke({"context": context, "question": query})
        logger.info(f"RAG Answer generated | confidence={response.confidence:.2f} | docs_used={len(docs)}")
        return response
    except Exception as e:
        logger.error(f"RAG Generation failed: {e}")
        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
        ])
        fallback_chain = fallback_prompt | get_llm()
        fallback_response = fallback_chain.invoke({"context": context, "question": query})
        return RAGResponse(
            answer=fallback_response.content,
            citations=[d.metadata.get("source", "unknown") for d in docs[:3]],
            confidence=0.5,
            has_sufficient_context=True,
        )
