import logging
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from config.settings import settings
from config.LLM_Factory import get_llm

logger = logging.getLogger(__name__)


class EntitySearchInput(BaseModel):
    entity_name: str = Field(..., description="Entity Name to search for in the graph")
    hops: int = Field(default=2, description="Number of relationship hops to traverse", ge=1, le=4)
    limit: int = Field(default=10, description="Max number of chunks to return", ge=1, le=50)


class CypherQueryInput(BaseModel):
    cypher: str = Field(..., description="Raw Cypher Query to Execute (READ only)")


class GraphRetriever:
    def __init__(self):
        self.driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        self._ensure_fulltext_index()

    def close(self):
        self.driver.close()

    def _ensure_fulltext_index(self):
        with self.driver.session() as session:
            try:
                session.run("""CREATE FULLTEXT INDEX chunk_text_index IF NOT EXISTS
                            FOR (c:Chunk) ON EACH [c.text]""")
            except Exception as e:
                logger.debug(f"Full-text index note: {e}")

    def entity_search(self, inp: EntitySearchInput) -> List[Document]:
        query = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($entity_name)
                WITH e LIMIT 5
                CALL apoc.path.subgraphNodes(e, {relationshipFilter: 'RELATED_TO', maxLevel: $hops})
                YIELD node AS neighbour
                WHERE neighbour:Entity
                MATCH (c:Chunk)-[:MENTIONS]->(neighbour)
                RETURN DISTINCT c.text AS text, c.chunk_index AS idx,
                neighbour.name AS entity, neighbour.type AS entity_type
                LIMIT $limit
                """
        try:
            with self.driver.session() as session:
                result = session.run(query, entity_name=inp.entity_name, hops=inp.hops, limit=inp.limit)
                docs = []
                for record in result:
                    docs.append(Document(
                        page_content=record["text"],
                        metadata={
                            "chunk_index": record["idx"],
                            "entity": record["entity"],
                            "entity_type": record["entity_type"],
                            "retrieval_source": "graph"
                        }
                    ))
                return docs
        except Exception as e:
            logger.warning(f"APOC entity search failed, falling back: {e}")
            return self.simple_entity_search(inp)

    def simple_entity_search(self, inp: EntitySearchInput) -> List[Document]:
        query = """MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($entity_name)
                RETURN DISTINCT c.text AS text, c.chunk_index as idx, e.name AS entity LIMIT $limit"""
        with self.driver.session() as session:
            result = session.run(query, entity_name=inp.entity_name, limit=inp.limit)
            return [Document(page_content=r["text"], metadata={"chunk_index": r["idx"], "entity": r["entity"], "retrieval_source": "graph"})
                    for r in result]

    def fulltext_search(self, query_text: str, limit: int = 10) -> List[Document]:
        query = """
                CALL db.index.fulltext.queryNodes('chunk_text_index',$query)
                YIELD node, score
                RETURN node.text as text, node.chunk_index as idx, score 
                ORDER BY score DESC
                LIMIT $limit"""
        with self.driver.session() as session:
            result = session.run(query, query=query_text, limit=limit)
            docs = []
            for record in result:
                docs.append(Document(page_content=record["text"], metadata={"chunk_index": record["idx"], "score": record["score"], "retrieval_source": "graph_fulltext"}))
            return docs

    def get_entity_relationships(self, entity_name: str) -> List[Dict[str, Any]]:
        query = """MATCH (e:Entity)-[r:RELATED_TO]->(t:Entity)
                WHERE toLower(e.name) CONTAINS toLower($name)
                RETURN e.name AS source, r.relation AS relation, t.name AS target
                LIMIT 20"""
        with self.driver.session() as session:
            result = session.run(query, name=entity_name)
            return [dict(r) for r in result]

    def execute_cypher(self, inp: CypherQueryInput) -> List[Dict[str, Any]]:
        cypher = inp.cypher.strip()
        if any(kw in cypher.upper() for kw in ["CREATE", "DELETE", "MERGE", "SET", "REMOVE", "DROP"]):
            raise ValueError("Only READ-only Cypher queries are allowed")
        with self.driver.session() as session:
            result = session.run(cypher)
            return [dict(r) for r in result]

    def nl_to_cypher_search(self, nl_query: str, limit: int = 10) -> List[Document]:
        schema_hint = """Nodes: (:Document {id, source, title}), (:Chunk {id, text, chunk_index}),
               (:Entity {id, name, type, description})
                Relationships: (:Document)-[:HAS_CHUNK]->(:Chunk),
                       (:Chunk)-[:MENTIONS]->(:Entity),
                       (:Entity)-[:RELATED_TO {relation}]->(:Entity)"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a Neo4j Cypher expert.
            Generate a READ-ONLY Cypher query to answer the question.
            Return ONLY the Cypher query, no explanation.
            Graph schema:
            {schema_hint}
            Always end with LIMIT {limit}."""),
            ("human", "{question}"),
        ])
        llm = get_llm()
        chain = prompt | llm
        try:
            response = chain.invoke({"question": nl_query})
            cypher = response.content.strip().strip("'''").replace("cypher", "").strip()
            logger.info(f"Generated Cypher: {cypher}")
            rows = self.execute_cypher(CypherQueryInput(cypher=cypher))
            docs = []
            for row in rows:
                text = row.get("text") or row.get("c.text") or str(row)
                docs.append(Document(page_content=text, metadata={"retrieval_source": "graph_nl", **row}))
            return docs
        except Exception as e:
            logger.warning(f"NL→Cypher failed: {e}")
            return self.fulltext_search(nl_query, limit=limit)


_retriever_instance: GraphRetriever | None = None


def get_graph_retrieval() -> GraphRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = GraphRetriever()
    return _retriever_instance


get_graph_retriever = get_graph_retrieval
