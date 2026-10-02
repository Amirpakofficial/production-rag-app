from functools import lru_cache
from langchain_chroma import Chroma
from app.core.config import get_settings
from app.services.embeddings import get_embedding_model
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache()
def get_vectorstore() -> Chroma:
    settings = get_settings()
    logger.info(f"Connecting to ChromaDB: {settings.chroma_persist_dir}")

    vectorstore = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=settings.chroma_persist_dir,
    )

    logger.info(f"ChromaDB ready. Chunks: {vectorstore._collection.count()}")
    return vectorstore


def reset_vectorstore() -> None:
    settings = get_settings()
    logger.warning("Resetting ChromaDB collection...")

    vs = get_vectorstore()
    vs._client.delete_collection(settings.chroma_collection_name)

    get_vectorstore.cache_clear()
    get_embedding_model.cache_clear()

    logger.info("ChromaDB reset complete.")


def get_document_count() -> int:
    try:
        return get_vectorstore()._collection.count()
    except Exception:
        return 0
