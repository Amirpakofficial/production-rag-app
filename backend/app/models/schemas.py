from pydantic import BaseModel, Field
from typing import Optional


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None


class SourceDocument(BaseModel):
    content: str
    source: str
    page: Optional[int] = None


class IngestResponse(BaseModel):
    message: str
    files_processed: int
    chunks_added: int


class StatusResponse(BaseModel):
    status: str
    document_count: int
    collection_name: str


class ResetResponse(BaseModel):
    message: str
