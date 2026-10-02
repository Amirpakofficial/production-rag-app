from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache()
def get_embedding_model() -> HuggingFaceEmbeddings:
    settings = get_settings()
    logger.info(f"Loading embedding model: {settings.embedding_model}")

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Embedding model loaded.")
    return embeddings
