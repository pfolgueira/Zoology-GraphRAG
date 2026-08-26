from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # Neo4j
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # Embeddings
    openrouter_api_key: str
    openrouter_embedding_model: str = "qwen/qwen3-embedding-8b"
    embedding_dimensions: int = 1024

    # LLM
    # openrouter_model: str

    # Reranking
    openrouter_rerank_model: str = "cohere/rerank-4-fast"

    # Processing
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_candidates: int = 15
    top_k_results: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

class GeminiSettings(BaseSettings):

    # Neo4j
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # Gemini
    gemini_api_key: str
    gemini_model: str = Field(
        validation_alias="GEMINI_MODEL"
    )

    # Processing
    chunk_size: int = 800
    chunk_overlap: int = 80
    top_k_results: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


class GroqSettings(BaseSettings):

    # Neo4j
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    # Groq
    groq_api_key: str
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias="GROQ_MODEL"
    )

    # Processing
    chunk_size: int = 800
    chunk_overlap: int = 80
    top_k_results: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()


@lru_cache()
def get_gemini_settings() -> GeminiSettings:
    return GeminiSettings()


@lru_cache()
def get_groq_settings() -> GroqSettings:
    return GroqSettings()