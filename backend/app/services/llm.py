from langchain_openai import ChatOpenAI
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_llm(
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    streaming: bool = False,
) -> ChatOpenAI:
    settings = get_settings()

    return ChatOpenAI(
        api_key=api_key or settings.llm_api_key,
        base_url=base_url or settings.llm_base_url,
        model=model_name or settings.llm_model_name,
        temperature=0.3,
        streaming=streaming,
        max_tokens=1024,
    )
