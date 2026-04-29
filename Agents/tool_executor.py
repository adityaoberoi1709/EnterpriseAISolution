import logging
from typing import List, Dict, Any

from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from config.LLM_Factory import get_llm
from Retrieval.hybrid_router import get_router
from Retrieval.graph_retrieval import get_graph_retrieval, EntitySearchInput, CypherQueryInput

logger = logging.getLogger(__name__)


# ── Schemas ───────────────────────────────────────────────────────────────────

class DocumentSearchInput(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="hybrid", description="Retrieval mode: vector, graph, hybrid")


class EntityLookupInput(BaseModel):
    entity_name: str = Field(..., description="Entity to look up e.g. 'TechNova', 'Sarah Chen', 'Bangalore'")
    hops: int = Field(default=2, ge=1, le=4, description="Graph traversal depth")


class RelationshipExploreInput(BaseModel):
    entity_name: str = Field(..., description="Entity to explore relationships for")


class GraphQueryInput(BaseModel):
    cypher_query: str = Field(..., description="READ-only Cypher query")


class MetadataFilterInput(BaseModel):
    source_filter: str | None = Field(None, description="Filter by source document name")
    limit: int = Field(default=10, ge=1, le=50)


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool("document_search", args_schema=DocumentSearchInput)
def document_search(query: str, top_k: int = 5, mode: str = "hybrid") -> str:
    """
    Search the enterprise knowledge base using hybrid vector + graph retrieval.
    Use this for broad questions, summaries, or when other tools return nothing.
    """
    try:
        router = get_router()
        docs = router.retrieve(query, mode=mode, top_k=top_k)
        if not docs:
            return "No relevant documents found. Try a different query."
        result = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "unknown")
            result.append(f"[{i}] ({src})\n{doc.page_content[:500]}")
        return "\n\n".join(result)
    except Exception as e:
        return f"document_search failed: {e}"


@tool("entity_lookup", args_schema=EntityLookupInput)
def entity_lookup(entity_name: str, hops: int = 2) -> str:
    """
    Look up a specific named entity in the Neo4j knowledge graph.
    Use for questions about specific people, organisations, places, or products.
    """
    try:
        retriever = get_graph_retrieval()
        docs = retriever.entity_search(EntitySearchInput(entity_name=entity_name, hops=hops, limit=8))
        if not docs:
            docs = retriever.fulltext_search(entity_name, limit=5)
        if not docs:
            return f"No graph entries found for '{entity_name}'. Try document_search."
        results = [f"Entity: {doc.metadata.get('entity', entity_name)} | {doc.page_content[:400]}" for doc in docs]
        return "\n\n".join(results)
    except Exception as e:
        return f"entity_lookup failed: {e}"


@tool("explore_relationships", args_schema=RelationshipExploreInput)
def explore_relationships(entity_name: str) -> str:
    """
    Return all known relationships for an entity from the knowledge graph.
    Use this when asked about connections or relationships between entities.
    For 'relationship between X and Y', call this for BOTH X and Y separately.
    """
    try:
        retriever = get_graph_retrieval()
        rels = retriever.get_entity_relationships(entity_name)
        if not rels:
            docs = retriever.fulltext_search(entity_name, limit=3)
            if docs:
                return (
                    f"No direct graph relationships for '{entity_name}', but found context:\n\n"
                    + "\n\n".join(d.page_content[:400] for d in docs)
                )
            return f"No relationships found for '{entity_name}'. Try entity_lookup or document_search."
        lines = [f"  {r['source']} --[{r['relation']}]--> {r['target']}" for r in rels]
        return f"Relationships for '{entity_name}':\n" + "\n".join(lines)
    except Exception as e:
        return f"explore_relationships failed: {e}"


@tool("run_graph_query", args_schema=GraphQueryInput)
def run_graph_query(cypher_query: str) -> str:
    """
    Execute a READ-only Cypher query against Neo4j.
    Use when you need precise structured lookups or want to build a custom query.
    """
    try:
        retriever = get_graph_retrieval()
        results = retriever.execute_cypher(CypherQueryInput(cypher=cypher_query))
        if not results:
            return "Query returned no results."
        return "\n".join(str(r) for r in results[:20])
    except ValueError as e:
        return f"Query rejected (write ops not allowed): {e}"
    except Exception as e:
        return f"run_graph_query failed: {e}"


@tool("filter_documents", args_schema=MetadataFilterInput)
def filter_documents(source_filter: str | None = None, limit: int = 10) -> str:
    """
    Filter documents in the knowledge base by source name.
    Useful for narrowing scope before a deeper search.
    """
    try:
        from Retrieval.vector_store import get_vector_store
        store = get_vector_store()
        meta_filter = {"source": source_filter} if source_filter else None
        docs = store.similarity_search("", k=limit, filter=meta_filter)
        if not docs:
            return "No documents match the filter."
        lines = [f"• {d.metadata.get('source', '?')} | chunk {d.metadata.get('chunk_index', '?')}" for d in docs]
        return "\n".join(lines)
    except Exception as e:
        return f"filter_documents failed: {e}"


ALL_TOOLS = [
    document_search,
    entity_lookup,
    explore_relationships,
    run_graph_query,
    filter_documents,
]

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an enterprise AI assistant with access to a hybrid knowledge base (vector store + Neo4j graph).

You have these tools:
- document_search: broad semantic search — use first for general questions
- entity_lookup: look up a specific named entity (person, org, place, product)
- explore_relationships: find how entities are connected — use for ANY relationship question
- run_graph_query: run a custom Cypher query on Neo4j
- filter_documents: filter by source document

STRICT RULES:
1. ALWAYS use at least one tool before answering. Never answer from memory.
2. For "relationship between X and Y" questions:
   - Call explore_relationships(X), then explore_relationships(Y), then document_search for context
3. If a tool returns empty, try another tool with a different query. Never give up after one empty result.
4. Only give your final answer after tools have returned data.
5. Always cite sources in your answer."""


# ── Agent ─────────────────────────────────────────────────────────────────────

def build_tool_agent():
    llm = get_llm()
    # Inject system prompt — compatible across all LangGraph versions
    try:
        # LangGraph >= 0.2.x
        return create_react_agent(llm, ALL_TOOLS, state_modifier=SYSTEM_PROMPT)
    except TypeError:
        try:
            # LangGraph 0.1.x
            return create_react_agent(llm, ALL_TOOLS, messages_modifier=[SystemMessage(content=SYSTEM_PROMPT)])
        except TypeError:
            # Oldest versions — no prompt support, still works, just no system prompt
            return create_react_agent(llm, ALL_TOOLS)


_tool_agent = None


def get_tool_agent():
    global _tool_agent
    if _tool_agent is None:
        _tool_agent = build_tool_agent()
    return _tool_agent


def run_tool_agent(query: str) -> Dict[str, Any]:
    # FIX 2: Reset agent per call so lru_cache on get_llm() doesn't
    # serve a stale provider after .env changes
    global _tool_agent
    _tool_agent = None
    agent = get_tool_agent()

    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config={"recursion_limit": 50},
    )

    messages = result.get("messages", [])

    # FIX 3: Find the real final answer — last AIMessage with content but NO tool_calls
    final_answer = ""
    for msg in reversed(messages):
        if (
            isinstance(msg, AIMessage)
            and msg.content
            and not getattr(msg, "tool_calls", None)
        ):
            final_answer = msg.content
            break

    tool_calls = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tool_calls.append(f"{tc['name']}({tc.get('args', {})})")

    return {
        "answer": final_answer or "Agent completed but produced no final answer.",
        "tool_calls": tool_calls,
        "messages": [m.content for m in messages if hasattr(m, "content") and m.content],
    }
