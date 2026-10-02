import os
import hashlib
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from app.core.config import get_settings
from app.services.vectorstore import get_vectorstore
from app.core.logging import get_logger

logger = get_logger(__name__)


def _safe_id(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text)


def _hash_content(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


def _load_file(file_path: str) -> list[Document]:
    ext = Path(file_path).suffix.lower()
    filename = Path(file_path).name

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = filename
        doc.metadata.setdefault("page", 0)

    return docs


def _chunk_documents(docs: list[Document]) -> list[Document]:
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(docs)
    logger.info(f"Split {len(docs)} pages into {len(chunks)} chunks.")
    return chunks


def ingest_files(file_paths: list[str]) -> dict:
    vectorstore = get_vectorstore()
    all_chunks: list[Document] = []

    for path in file_paths:
        try:
            docs = _load_file(path)
            chunks = _chunk_documents(docs)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")
            continue

    if not all_chunks:
        return {"files_processed": 0, "chunks_added": 0}

    ids = [
        _safe_id(f"{c.metadata.get('source', 'doc')}_{_hash_content(c.page_content)}")
        for c in all_chunks
    ]

    vectorstore.add_documents(documents=all_chunks, ids=ids)
    logger.info(f"Indexed {len(all_chunks)} chunks.")

    return {
        "files_processed": len(file_paths),
        "chunks_added": len(all_chunks),
    }
