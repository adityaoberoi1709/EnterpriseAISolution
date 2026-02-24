import hashlib
import logging 
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from config.settings import settings
from config.LLM_Factory import get_llm

logger = logging.getLogger(__name__)


class Entity(BaseModel):
    name: str = Field(..., description="Entity Name")
    type: str = Field(..., description="Entity Type: PERSON, ORG, CONCEPT, PRODUCT, LOCATION")
    description: str = Field(default="", description="Short description")


class Relationship(BaseModel):
    source: str = Field(..., description="Source Entity Name")
    target: str = Field(..., description="Target Entity Name")
    relation: str = Field(..., description="Relationship Label")


class GraphExtractionResult(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)


EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a knowledge graph extraction expert.
Extract all named entities and relationships from the text.
Return ONLY valid JSON matching this schema:
{{
  "entities": [{{"name": str, "type": str, "description": str}}],
  "relationships": [{{"source": str, "target": str, "relation": str}}]
}}
Entity types: PERSON, ORG, CONCEPT, PRODUCT, LOCATION, TECHNOLOGY
Keep entity names concise and consistent."""),
    ("user", "Extract entities and relationships from:\n\n{text}"),
])


def extract_graph_elements(text: str) -> GraphExtractionResult:
    llm = get_llm()
    structured_llm = llm.with_structured_output(GraphExtractionResult)
    chain = EXTRACTION_PROMPT | structured_llm
    try:
        result = chain.invoke({"text": text[:2000]})
        return result
    except Exception as e:
        logger.warning(f"Graph extraction failed: {e}")
        return GraphExtractionResult()


class Neo4jGraphBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        self._ensure_constraints()

    def close(self):
        self.driver.close()

    def _ensure_constraints(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")

    def _upsert_document(self, session, doc_id: str, source: str, title: str):
        session.run(
            "MERGE (d:Document {id: $id}) SET d.source = $source, d.title = $title",
            id=doc_id, source=source, title=title
        )

    def _upsert_chunk(self, session, chunk_id: str, text: str, index: int, doc_id: str):
        session.run(
            """MERGE (c:Chunk {id: $id})
            SET c.text = $text, c.chunk_index = $index
            WITH c MATCH (d:Document {id: $doc_id})
            MERGE (d)-[:HAS_CHUNK]->(c)""",
            id=chunk_id, text=text, index=index, doc_id=doc_id
        )

    def _upsert_entity(self, session, entity: Entity, chunk_id: str):
        ent_id = hashlib.md5(entity.name.lower().encode()).hexdigest()
        session.run(
            """MERGE (e:Entity {id: $id})
              SET e.name = $name, e.type = $type, e.description = $description
              WITH e MATCH (c:Chunk {id: $chunk_id})
              MERGE (c)-[:MENTIONS]->(e)""",
            id=ent_id,
            name=entity.name,
            type=entity.type,
            description=entity.description,
            chunk_id=chunk_id
        )
        return ent_id

    def _upsert_relationship(self, session, rel: Relationship):
        src_id = hashlib.md5(rel.source.lower().encode()).hexdigest()
        tgt_id = hashlib.md5(rel.target.lower().encode()).hexdigest()
        session.run(
            """MERGE (s:Entity {id: $src})
              MERGE (t:Entity {id: $tgt})
              MERGE (s)-[r:RELATED_TO {relation: $relation}]->(t)""",
            src=src_id, tgt=tgt_id, relation=rel.relation
        )

    def build_from_chunks(self, chunks: List[Document]) -> None:
        with self.driver.session() as session:
            for chunk in chunks:
                source = chunk.metadata.get("source", "unknown")
                doc_id = hashlib.md5(source.encode()).hexdigest()
                chunk_id = hashlib.md5(
                    (source + str(chunk.metadata.get("chunk_index", 0))).encode()
                ).hexdigest()
                self._upsert_document(session, doc_id, source, source.split("/")[-1])
                self._upsert_chunk(session, chunk_id, chunk.page_content, chunk.metadata.get("chunk_index", 0), doc_id)
                extraction = extract_graph_elements(chunk.page_content)
                for entity in extraction.entities:
                    self._upsert_entity(session, entity, chunk_id)
                for rel in extraction.relationships:
                    self._upsert_relationship(session, rel)
        logger.info("Knowledge Graph Build Complete")

    def get_stats(self) -> dict:
        with self.driver.session() as session:
            counts = {}
            for label in ["Document", "Chunk", "Entity"]:
                result = session.run(f"Match (n:{label}) return count(n) as c")
                counts[label] = result.single()["c"]
            rel_result = session.run("MATCH ()-[r:RELATED_TO]->() return count(r) as c")
            counts["Relationship"] = rel_result.single()["c"]
        return counts
