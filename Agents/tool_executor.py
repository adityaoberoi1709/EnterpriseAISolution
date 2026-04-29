import logging
from typing import List, Dict, Any

from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from config.LLM_Factory import get_llm
from config.settings import settings
from Retrieval.hybrid_router import get_router
from Retrieval.graph_retrieval import get_graph_retrieval, EntitySearchInput, CypherQueryInput

logger = logging.getLogger(__name__)


class DocumentSearchInput(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="hybrid", description="Retrieval mode: vector, graph, hybrid")


class EntityLookupInput(BaseModel):
    entity_name: str = Field(..., description="Entity to look up")
    hops: int = Field(default=2, ge=1, le=4, description="Graph traversal depth")


class RelationshipExploreInput(BaseModel):
    entity_name: str = Field(..., description="Entity to explore relationships for")


class GraphQueryInput(BaseModel):
    cypher_query: str = Field(..., description="READ-only Cypher query")


class MetadataFilterInput(BaseModel):
    source_filter: str | None = Field(None, description="Filter by source document name")
    entity_type_filter: str | None = Field(None, description="Filter by entity type: PERSON, ORG")
    limit: int = Field(default=10, ge=1, le=50)


@tool("document_search", args_schema=DocumentSearchInput)
def document_search(query: str, top_k: int = 5, mode: str = "hybrid") -> str:
    """
    Search the enterprise knowledge base using hybrid vector + graph retrieval.
    Returns the most relevant document chunks for any broad question.
    Use this first before trying more specific tools.
    """
    try:
        router = get_router()
        docs = router.retrieve(query, mode=mode, top_k=top_k)
        if not docs:
            return "No relevant documents found. Try a different query or use entity_lookup."
        result = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "unknown")
            result.append(f"[{i}] ({src})\n{doc.page_content[:600]}")
        return "\n\n".join(result)
    except Exception as e:
        logger.error(f"document_search failed: {e}")
        return f"Search failed: {e}. Try entity_lookup instead."


@tool("entity_lookup", args_schema=EntityLookupInput)
def entity_lookup(entity_name: str, hops: int = 2) -> str:
    """
    Look up a specific entity in the knowledge graph and return related context.
    Use this for questions about specific people, organisations, products, or locations.
    Example: entity_name='TechNova', entity_name='Bangalore', entity_name='Sarah Chen'
    """
    try:
        retriever = get_graph_retrieval()
        inp = EntitySearchInput(entity_name=entity_name, hops=hops, limit=8)
        docs = retriever.entity_search(inp)
        if not docs:
            docs = retriever.fulltext_search(entity_name, limit=5)
        if not docs:
            return f"No graph entries found for '{entity_name}'. Try document_search with a broader query."
        results = [f"Entity: {doc.metadata.get('entity', entity_name)} | {doc.page_content[:500]}" for doc in docs]
        return "\n\n".join(results)
    except Exception as e:
        logger.error(f"entity_lookup failed: {e}")
        return f"Entity lookup failed: {e}. Try document_search instead."


@tool("explore_relationships", args_schema=RelationshipExploreInput)
def explore_relationships(entity_name: str) -> str:
    """
    Return all known relationships for an entity from the knowledge graph.
    Use this to understand how two entities are connected, e.g. 'TechNova and Bangalore'.
    Always call this when the question asks about relationships, connections, or links between entities.
    """
    try:
        retriever = get_graph_retrieval()
        rels = retriever.get_entity_relationships(entity_name)
        if not rels:
            docs = retriever.fulltext_search(entity_name, limit=3)
            if docs:
                return (
                    f"No direct graph relationships found for '{entity_name}', "
                    f"but found relevant context:\n\n"
                    + "\n\n".join(d.page_content[:400] for d in docs)
                )
            return f"No relationships found for '{entity_name}'. Try entity_lookup or document_search."
        lines = [f"• {r['source']} —[{r['relation']}]→ {r['target']}" for r in rels]
        return f"Relationships for '{entity_name}':\n" + "\n".join(lines)
    except Exception as e:
        logger.error(f"explore_relationships failed: {e}")
        return f"Relationship exploration failed: {e}. Try entity_lookup instead."


@tool("run_graph_query", args_schema=GraphQueryInput)
def run_graph_query(cypher_query: str) -> str:
    """
    Execute a read-only Cypher query against the Neo4j knowledge graph.
    Useful for precise structured lookups when you know what to query.
    """
    try:
        retriever = get_graph_retrieval()
        results = retriever.execute_cypher(CypherQueryInput(cypher=cypher_query))
        if not results:
            return "Query returned no results."
        lines = [str(r) for r in results[:20]]
        return "\n".join(lines)
    except ValueError as e:
        return f"Query rejected (write operations not allowed): {e}"
    except Exception as e:
        return f"Query failed: {e}"


@tool("filter_documents", args_schema=MetadataFilterInput)
def filter_documents(
    source_filter: str | None = None,
    entity_type_filter: str | None = None,
    limit: int = 10,
) -> str:
    """
    Filter documents in the knowledge base by source name or entity type.
    Useful for narrowing scope before a deeper search.
    """
    try:
        from Retrieval.vector_store import get_vector_store
        store = get_vector_store()
        meta_filter = {}
        if source_filter:
            meta_filter["source"] = source_filter
        docs = store.similarity_search("", k=limit, filter=meta_filter if meta_filter else None)
        if not docs:
            return "No documents match the filter."
        lines = [f"• {d.metadata.get('source', '?')} | chunk {d.metadata.get('chunk_index', '?')}" for d in docs]
        return "\n".join(lines)
    except Exception as e:
        return f"Filter failed: {e}"


ALL_TOOLS = [
    document_search,
    entity_lookup,
    explore_relationships,
    run_graph_query,
    filter_documents,
]

SYSTEM_PROMPT = """You are an enterprise AI assistant with access to a hybrid knowledge base (vector store + knowledge graph).

IMPORTANT RULES:
1. ALWAYS use tools to find information before answering. Never answer from memory alone.
2. For relationship questions (e.g. "relation between X and Y"), call BOTH:
   - explore_relationships(entity_name="X")
   - explore_relationships(entity_name="Y")
   - Then document_search to find any additional context.
3. Do NOT stop after one tool call if the result is empty or partial. Try at least 2-3 tools before concluding.
4. If entity_lookup returns nothing, immediately fall back to document_search.
5. Only say "I don't know" after trying at least 2 different tools with 2 different queries.
6. Always cite your sources at the end of your answer.
"""


def build_tool_agent():
    llm = get_llm()
    return create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)


_tool_agent = None


def get_tool_agent():
    global _tool_agent
    if _tool_agent is None:
        _tool_agent = build_tool_agent()
    return _tool_agent


def run_tool_agent(query: str) -> Dict[str, Any]:
    global _tool_agent
    _tool_agent = None
    agent = get_tool_agent()

    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": 50},
    )

    messages = result.get("messages", [])
    final_answer = messages[-1].content if messages else "No answer generated"

    tool_calls = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(f"{tc['name']}({tc.get('args', {})})")

    return {
        "answer": final_answer,
        "tool_calls": tool_calls,
        "messages": [m.content for m in messages if hasattr(m, "content") and m.content],
    }
