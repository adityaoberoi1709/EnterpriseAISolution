# GraphAgentix

A production-grade Hybrid GraphRAG platform combining FAISS vector search, Neo4j knowledge graphs, LangGraph agents, and RAGAS evaluation.

## Architecture

```
api/main.py              ← FastAPI entry point
Agents/
  planner_agent.py       ← LangGraph multi-step planner
  rag_agent.py           ← RAG answer generation
  tool_executor.py       ← ReAct tool-calling agent
Ingestion/
  ingestion_chunking.py  ← Document loading & chunking
  vector_store.py        ← FAISS vector store
  GraphBuilder.py        ← Neo4j graph construction
  embedding.py           ← Embedding with Redis cache
Retrieval/
  hybrid_router.py       ← Hybrid vector+graph retrieval
  graph_retrieval.py     ← Neo4j graph queries
Evaluation/
  metrics.py             ← Latency, confidence, Prometheus
  ragas_eval.py          ← RAGAS faithfulness/relevancy eval
config/
  settings.py            ← Pydantic settings (reads .env)
  LLM_Factory.py         ← LLM / embedding factory
```

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Neo4j (local or cloud) — needed for graph features
- Redis (optional) — embedding cache

### 2. Set up environment

```bash
# Clone / unzip the project
cd EnterpriseAISolution

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure secrets

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (and NEO4J_PASSWORD etc.)
```

### 4. Run the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs for the interactive Swagger UI.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/ingest` | Ingest documents |
| POST | `/query` | RAG query |
| POST | `/agent` | Planner or tool agent |
| GET | `/graph/entity` | Entity graph lookup |
| POST | `/eval` | RAGAS evaluation |
| GET | `/metrics` | Prometheus-style metrics |
| GET | `/graph/stats` | Neo4j node/edge counts |

## Running in VS Code

1. Install the **Python** extension in VS Code.
2. Open the project folder: `File → Open Folder → EnterpriseAISolution`
3. Press `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the `venv` you created.
4. Open a terminal (`Ctrl+\``) — it will auto-activate the venv.
5. Run `uvicorn api.main:app --reload`
.
.
