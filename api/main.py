import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import settings
from Evaluation.metrics import (
    MetricsStore, 
    RequestMetrics,
    compute_source_diversity,
    compute_hallucination_risk
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Enterprise GenAI Platform...")
    try:
        from Ingestion.vector_store import get_vector_store
        from Retrieval.graph_retrieval import get_graph_retrieval
        get_vector_store()
        get_graph_retrieval()
        logger.info("Vector Store and graph retriever ready")
    except Exception as e:
        logger.warning(f"Could not pre-load stores : {e}")
    yield
    logger.info("Shutting down")
    MetricsStore.save()


app = FastAPI(
    title="Enterprise GenAI Platform",
    description="GraphRAG Backend - Hybrid, Langgraph agents, RAGAS Evaluation",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class IngestRequest(BaseModel):
    path: str = Field(..., description="File Directory")
    strategy: Literal["semantic", "recursive", "fixed"] = "recursive"
    build_graph: bool = True


class IngestResponse(BaseModel):
    chunks_created: int
    vectors_indexed: int
    graph_built: bool
    message: str


class QueryRequest(BaseModel):
    query: str
    mode: Literal["vector", "graph", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    confidence: float
    has_sufficient_context: bool
    retrieval_mode: str
    latency_ms: float


class AgentRequest(BaseModel):
    query: str
    thread_id: str = "default"
    use_tools: bool = False


class AgentResponse(BaseModel):
    answer: str
    plan: List[str] = []
    step_results: List[str] = []
    tool_calls: List[str] = []
    latency_ms: float


class EvalRequest(BaseModel):
    samples: List[dict] = Field(
        default=[],
        description="[{question, ground_truth}] — if empty uses built-in samples"
    )
    save_report: bool = True


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "provider": settings.LLM_PROVIDER,
        "model": settings.OPENAI_MODEL if settings.LLM_PROVIDER == "openai" else settings.ANTHROPIC_MODEL
    }


#We should ideally use track_latency
@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    from Ingestion.ingestion_chunking import ingest_path
    from Ingestion.vector_store import get_vector_store
    try:
        chunks = ingest_path(req.path, strategy=req.strategy)
        store = get_vector_store()
        store.build_from_documents(chunks)
        if req.build_graph:
            background_tasks.add_task(_build_graph_background, chunks)
        return IngestResponse(
            chunks_created=len(chunks),
            vectors_indexed=store.total_vectors,
            graph_built=req.build_graph,
            message=f"Successfully ingested {Path(req.path).name}"
        )
    except Exception as e:
        logger.error(f"Ingest Failed : {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _build_graph_background(chunks):
    from Ingestion.GraphBuilder import Neo4jGraphBuilder
    try:
        builder = Neo4jGraphBuilder()
        builder.build_from_chunks(chunks)
        builder.close()
    except Exception as e:
        logger.error(f"Background graph build failed : {e}")


#We should ideally use track_latency
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    from Agents.rag_agent import answer, retrieve_context
    start = time.perf_counter()
    try:
        rag_resp = answer(req.query, top_k=req.top_k, retrieval_mode=req.mode)
        docs = retrieve_context(req.query, top_k=req.top_k)
        latency = (time.perf_counter() - start) * 1000
        MetricsStore.record_request(RequestMetrics(
            query=req.query,
            retrieval_mode=req.mode,
            total_docs_retrieved=len(docs),
            total_latency_ms=latency,
            confidence=rag_resp.confidence,
            source_diversity=compute_source_diversity(docs),
            hallucination_risk=compute_hallucination_risk(rag_resp.answer, docs)
        ))
        return QueryResponse(
            answer=rag_resp.answer,
            citations=rag_resp.citations,
            confidence=rag_resp.confidence,
            has_sufficient_context=rag_resp.has_sufficient_context,
            retrieval_mode=req.mode,
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        logger.error(f"Query failed : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent", response_model=AgentResponse)
async def agent(req: AgentRequest):
    start = time.perf_counter()
    try:
        if req.use_tools:
            from Agents.tool_executor import run_tool_agent
            result = run_tool_agent(req.query)
            latency = (time.perf_counter() - start) * 1000
            return AgentResponse(
                answer=result["answer"],
                tool_calls=result["tool_calls"],
                latency_ms=round(latency, 2)
            )
        else:
            from Agents.planner_agent import run_planner
            result = run_planner(req.query, thread_id=req.thread_id)
            latency = (time.perf_counter() - start) * 1000
            return AgentResponse(
                answer=result["answer"],
                plan=result["plan"],
                step_results=result["step_results"],
                latency_ms=round(latency, 2)
            )
    except Exception as e:
        logger.error(f"Agent Failed : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/entity")
async def graph_entity(entity_name: str, hops: int = 2):
    from Retrieval.graph_retrieval import get_graph_retrieval, EntitySearchInput
    try:
        retriever = get_graph_retrieval()
        inp = EntitySearchInput(entity_name=entity_name, hops=hops)
        docs = retriever.entity_search(inp)
        rels = retriever.get_entity_relationships(entity_name)
        return {
            "entity": entity_name,
            "chunks_found": len(docs),
            "relationships": rels,
            "contexts": [{"text": d.page_content[:400], "meta": d.metadata} for d in docs]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/eval")
async def run_evaluation(req: EvalRequest, background_tasks: BackgroundTasks):
    from Evaluation.ragas_eval import EvalSample, evaluate_batch
    samples = [EvalSample(**s) for s in req.samples] if req.samples else _default_eval_samples()
    output_path = "data/eval_results.json" if req.save_report else None
    background_tasks.add_task(evaluate_batch, samples, output_path)
    return {
        "message": f"Evaluation started for {len(samples)} samples",
        "output": output_path,
    }


def _default_eval_samples():
    from Evaluation.ragas_eval import EvalSample
    return [
        EvalSample(question="What is the main topic of the documents?"),
        EvalSample(question="What entities are mentioned most frequently?"),
        EvalSample(question="Summarize the key information in the knowledge base"),
    ]


@app.get("/metrics")
async def get_metrics():
    return MetricsStore.get_summary()


@app.get("/graph/stats")
async def graph_stats():
    from Ingestion.GraphBuilder import Neo4jGraphBuilder
    try:
        builder = Neo4jGraphBuilder()
        stats = builder.get_stats()
        builder.close()
        return stats
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {e}")


if __name__ == "__main__":
    uvicorn.run("api.main:app",
                host=settings.API_HOST,
                port=settings.API_PORT,
                reload=settings.API_RELOAD,
                log_level="info")
