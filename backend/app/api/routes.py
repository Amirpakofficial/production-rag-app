import os
import aiofiles
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import (
    ChatRequest, IngestResponse, StatusResponse, ResetResponse
)
from app.services.ingestion import ingest_files
from app.services.rag import stream_rag_response
from app.services.vectorstore import reset_vectorstore, get_document_count
from app.core.config import get_settings
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@router.get("/status", response_model=StatusResponse)
async def get_status():
    settings = get_settings()
    return StatusResponse(
        status="ok",
        document_count=get_document_count(),
        collection_name=settings.chroma_collection_name,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    saved_paths = []
    for file in files:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file: {file.filename}. Only PDF and TXT allowed.",
            )

        save_path = os.path.join(UPLOAD_DIR, file.filename)
        async with aiofiles.open(save_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        saved_paths.append(save_path)
        logger.info(f"Saved: {file.filename}")

    try:
        result = ingest_files(saved_paths)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(
        message=f"Processed {result['files_processed']} file(s).",
        files_processed=result["files_processed"],
        chunks_added=result["chunks_added"],
    )


@router.post("/chat")
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    async def event_generator():
        try:
            async for token in stream_rag_response(
                question=request.question,
                history=[msg.model_dump() for msg in request.history],
                api_key=request.api_key,
                base_url=request.base_url,
                model_name=request.model_name,
            ):
                yield token
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={"X-Accel-Buffering": "no"},
    )


@router.post("/reset", response_model=ResetResponse)
async def reset_database():
    try:
        reset_vectorstore()
        return ResetResponse(message="Vector database cleared successfully.")
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
