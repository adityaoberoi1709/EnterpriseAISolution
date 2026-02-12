import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # LLM
    LLM_PROVIDER: str = Field(default="openai", description="openai or anthropic")
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic API key")
    AGENT_MODEL: str = Field(default="gpt-4o-mini")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")
    ANTHROPIC_MODEL: str = Field(default="claude-3-5-sonnet-20241022")
    AGENT_TEMPERATURE: float = Field(default=0.0)
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")
    EMBEDDING_BATCH_SIZE: int = Field(default=100)

    # FAISS
    FAISS_INDEX_PATH: str = Field(default="data/faiss_index")
    FAISS_TOP_K: int = Field(default=10)

    # Neo4j
    NEO4J_URI: str = Field(default="bolt://localhost:7687")
    NEO4J_USER: str = Field(default="neo4j")
    NEO4J_PASSWORD: str = Field(default="password")

    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379")
    REDIS_CACHE_TTL: int = Field(default=3600)

    # Chunking
    CHUNK_SIZE: int = Field(default=512)
    CHUNK_OVERLAP: int = Field(default=64)
    CHUNKING_STRATEGY: str = Field(default="recursive")

    # Retrieval
    RERANKER_TOP_N: int = Field(default=5)
    HYBRID_ALPHA: float = Field(default=0.7)

    # Agent
    MAX_AGENT_ITERATION: int = Field(default=6)

    # API
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_RELOAD: bool = Field(default=False)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
