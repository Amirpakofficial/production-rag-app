import json
from typing import AsyncGenerator

from langchain.schema import Document, HumanMessage, AIMessage, SystemMessage

from app.services.vectorstore import get_vectorstore
from app.services.llm import get_llm
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context documents.

Rules:
- Answer using only the information found in the context below.
- If the context does not contain the answer, say: "I don't have enough information in the uploaded documents to answer that."
- Be concise and accurate.
- Cite the source filename when referencing specific information.

Context:
{context}
"""


def _format_history(history: list[dict]) -> list:
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        page_str = f", page {page}" if page else ""
        parts.append(f"[{i}] Source: {source}{page_str}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def retrieve_documents(question: str) -> list[Document]:
    settings = get_settings()
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(question, k=settings.retrieval_top_k)
    logger.info(f"Retrieved {len(docs)} chunks for: '{question[:60]}'")
    return docs


async def stream_rag_response(
    question: str,
    history: list[dict],
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
) -> AsyncGenerator[str, None]:
    docs = retrieve_documents(question)

    if not docs:
        yield "I don't have any documents to search through. Please upload some files first."
        return

    context_text = _format_context(docs)
    system_message = SystemMessage(content=SYSTEM_PROMPT.format(context=context_text))

    messages = [system_message]
    messages.extend(_format_history(history))
    messages.append(HumanMessage(content=question))

    llm = get_llm(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        streaming=True,
    )

    async for chunk in llm.astream(messages):
        if chunk.content:
            yield str(chunk.content)

    sources = [
        {
            "content": doc.page_content[:200] + "...",
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", None),
        }
        for doc in docs
    ]
    yield f"\n__SOURCES__:{json.dumps(sources)}"
