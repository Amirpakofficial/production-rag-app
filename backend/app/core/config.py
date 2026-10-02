from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_api_key: str = "sk-placeholder"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model_name: str = "gpt-4o-mini"

    embedding_model: str = "all-MiniLM-L6-v2"

    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "rag_documents"

    chunk_size: int = 1000
    chunk_overlap: int = 200

    retrieval_top_k: int = 4

    backend_host: str = "0.0.0.0"
    backend_port: int = 8000


@lru_cache()
def get_settings() -> Settings:
    return Settings()
