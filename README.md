# RAG Chatbot

A simple, production-ready RAG (Retrieval-Augmented Generation) chatbot. Upload your PDF or TXT files and chat with them using any OpenAI-compatible LLM.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-red)
![LangChain](https://img.shields.io/badge/LangChain-0.3-yellow)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-purple)

---

## Features

- Upload PDF and TXT files (multi-file support)
- Local embeddings with `all-MiniLM-L6-v2` — no extra API needed
- Streaming chat responses token by token
- Source citations with every answer
- Persistent ChromaDB vector store
- Works with any OpenAI-compatible API (OpenAI, Groq, Together AI, Ollama)
- Clean dark UI similar to ChatGPT
- Session history with reset controls

---

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + uvicorn |
| Frontend | Streamlit |
| LLM | LangChain + ChatOpenAI (any compatible API) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (persistent) |
| Chunking | RecursiveCharacterTextSplitter |

---

## Project Structure

```
rag-chatbot/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── logging.py
│   │   ├── services/
│   │   │   ├── embeddings.py
│   │   │   ├── vectorstore.py
│   │   │   ├── ingestion.py
│   │   │   ├── llm.py
│   │   │   └── rag.py
│   │   ├── models/
│   │   │   └── schemas.py
│   │   └── main.py
│   ├── chroma_db/
│   ├── uploads/
│   └── requirements.txt
├── frontend/
│   ├── app.py
│   └── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
LLM_API_KEY=sk-your-key-here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini
```

> The API key can also be entered directly in the sidebar UI at runtime.

### 3. Install backend

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Pre-download the embedding model (recommended)

Run this once before starting the server to avoid timeout on first upload:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### 5. Install frontend

```bash
cd ../frontend
pip install -r requirements.txt
```

---

## Run

**Terminal 1 — Backend:**

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend:**

```bash
cd frontend
streamlit run app.py
```

Open **http://localhost:8501**

API docs: **http://localhost:8000/docs**

---

## Usage

1. Enter your API key and model in the sidebar (or set in `.env`)
2. Upload PDF or TXT files
3. Click **Index Documents** and wait for confirmation
4. Start chatting — answers come only from your uploaded documents
5. Use **Clear Chat** to reset history, **Reset DB** to clear all documents

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/status` | DB status and chunk count |
| POST | `/api/ingest` | Upload and index files |
| POST | `/api/chat` | Streaming RAG response |
| POST | `/api/reset` | Clear vector database |
| GET | `/docs` | Swagger UI |

---

## Supported LLM Providers

| Provider | Base URL |
|---|---|
| OpenAI | `https://api.openai.com/v1` |
| Groq | `https://api.groq.com/openai/v1` |
| Together AI | `https://api.together.xyz/v1` |
| Ollama (local) | `http://localhost:11434/v1` |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_API_KEY` | `sk-placeholder` | API key |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | LLM endpoint |
| `LLM_MODEL_NAME` | `gpt-4o-mini` | Model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `CHUNK_SIZE` | `1000` | Chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `4` | Chunks retrieved per query |

---

## License

MIT
