from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from config.settings import settings

@lru_cache(maxsize=1)
def get_llm(temperature: float | None = None) -> BaseChatModel:
    temp = temperature if temperature is not None else settings.AGENT_TEMPERATURE
    if settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=settings.AGENT_MODEL, api_key=settings.OPENAI_API_KEY, temperature=temp)
    if settings.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=settings.AGENT_MODEL, api_key=settings.ANTHROPIC_API_KEY, temperature=temp)
    raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")

@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)

@lru_cache(maxsize=1)
def get_embedding_model():
    return get_embeddings()
