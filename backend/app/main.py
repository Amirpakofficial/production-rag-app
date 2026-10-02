from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title="RAG Chatbot API",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    logger.info("RAG Chatbot API started")
    logger.info(f"Model : {settings.llm_model_name}")
    logger.info(f"Embed : {settings.embedding_model}")
    logger.info(f"DB    : {settings.chroma_persist_dir}")


@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.backend_host, port=settings.backend_port, reload=True)
