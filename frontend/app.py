import json
import requests
import streamlit as st

st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://localhost:8000/api"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d0d0d !important;
    color: #e8e8e8 !important;
    font-family: 'Syne', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding-top: 1rem !important; }

[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid #222 !important;
}
[data-testid="stSidebar"] * { color: #c8c8c8 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f0f0f0 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
}

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
}

.source-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin: 0.3rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #888 !important;
}
.source-card strong { color: #4f8ef7 !important; }

[data-testid="stChatInput"] {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    color: #e8e8e8 !important;
    font-family: 'Syne', sans-serif !important;
}

.stButton > button {
    background: #1a1a1a !important;
    color: #c8c8c8 !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #242424 !important;
    border-color: #555 !important;
    color: #fff !important;
}

[data-testid="stFileUploader"] {
    background: #141414 !important;
    border: 1px dashed #333 !important;
    border-radius: 10px !important;
}

[data-testid="stTextInput"] input {
    background: #1a1a1a !important;
    color: #e8e8e8 !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    color: #888;
    padding: 4px 10px;
    background: #141414;
    border-radius: 20px;
    border: 1px solid #222;
    margin-bottom: 1rem;
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: #27c93f; }

hr { border-color: #222 !important; margin: 1rem 0 !important; }

[data-testid="stExpander"] {
    background: #141414 !important;
    border: 1px solid #222 !important;
    border-radius: 8px !important;
}

.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #f0f0f0;
    letter-spacing: -0.02em;
    margin-bottom: 0.1rem;
}
.app-subtitle {
    font-size: 0.8rem;
    color: #555;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 1.2rem;
}
</style>
""", unsafe_allow_html=True)


def init_session():
    defaults = {
        "messages": [],
        "doc_count": 0,
        "llm_api_key": "",
        "llm_base_url": "https://api.openai.com/v1",
        "llm_model": "gpt-4o-mini",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session()


def fetch_status() -> dict:
    try:
        r = requests.get(f"{BACKEND_URL}/status", timeout=5)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def upload_files(files) -> dict:
    file_tuples = [("files", (f.name, f.getvalue(), f.type)) for f in files]
    try:
        r = requests.post(f"{BACKEND_URL}/ingest", files=file_tuples, timeout=300)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def stream_chat(question: str, history: list):
    payload = {
        "question": question,
        "history": history,
        "api_key": st.session_state.llm_api_key or None,
        "base_url": st.session_state.llm_base_url or None,
        "model_name": st.session_state.llm_model or None,
    }

    full_text = ""
    sources = []

    with requests.post(
        f"{BACKEND_URL}/chat",
        json=payload,
        stream=True,
        timeout=60,
    ) as response:
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                if "__SOURCES__:" in chunk:
                    parts = chunk.split("__SOURCES__:", 1)
                    full_text += parts[0]
                    try:
                        sources = json.loads(parts[1])
                    except Exception:
                        pass
                else:
                    full_text += chunk
                    yield full_text, sources

    yield full_text, sources


def reset_database() -> str:
    try:
        r = requests.post(f"{BACKEND_URL}/reset", timeout=10)
        return r.json().get("message", "Done.")
    except Exception as e:
        return f"Error: {e}"


with st.sidebar:
    st.markdown('<div class="app-title">RAG Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">retrieval-augmented generation</div>', unsafe_allow_html=True)

    status = fetch_status()
    if status:
        doc_count = status.get("document_count", 0)
        st.session_state.doc_count = doc_count
        st.markdown(
            f'<div class="status-badge"><span class="status-dot"></span>'
            f'{doc_count} chunks indexed</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge" style="border-color:#7f1d1d">'
            '<span class="status-dot" style="background:#f87171"></span>'
            'Backend offline</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("#### Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("Index Documents", use_container_width=True):
            with st.spinner("Processing... (first run downloads the embedding model ~90MB)"):
                result = upload_files(uploaded_files)
            if "error" in result:
                st.error(f"Failed: {result['error']}")
            else:
                st.success(
                    f"{result.get('files_processed', 0)} file(s) · "
                    f"{result.get('chunks_added', 0)} chunks added"
                )
                st.rerun()

    st.markdown("---")

    st.markdown("#### LLM Settings")
    st.session_state.llm_api_key = st.text_input(
        "API Key", value=st.session_state.llm_api_key,
        type="password", placeholder="sk-...",
    )
    st.session_state.llm_base_url = st.text_input(
        "Base URL", value=st.session_state.llm_base_url,
        placeholder="https://api.openai.com/v1",
    )
    st.session_state.llm_model = st.text_input(
        "Model Name", value=st.session_state.llm_model,
        placeholder="gpt-4o-mini",
    )
    st.caption("Supports OpenAI, Groq, Together AI, Ollama")

    st.markdown("---")

    st.markdown("#### Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("Reset DB", use_container_width=True):
            with st.spinner("Clearing..."):
                msg = reset_database()
            st.success(msg)
            st.rerun()


st.markdown("### Ask your documents")
st.caption(
    f"{str(st.session_state.doc_count) + ' chunks ready' if st.session_state.doc_count else 'No documents indexed — upload files in the sidebar'}"
)
st.markdown("---")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"{len(msg['sources'])} source(s)", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'<strong>{src["source"]}</strong>'
                        f'{(" · page " + str(src["page"])) if src.get("page") else ""}'
                        f'<br><span style="color:#666">{src["content"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

if prompt := st.chat_input("Ask something about your documents…"):
    if st.session_state.doc_count == 0:
        st.warning("Please upload and index documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_answer = ""
            final_sources = []

            try:
                for current_text, current_sources in stream_chat(prompt, history):
                    full_answer = current_text
                    final_sources = current_sources
                    placeholder.markdown(full_answer + "▌")

                placeholder.markdown(full_answer)

                if final_sources:
                    with st.expander(f"{len(final_sources)} source(s)", expanded=False):
                        for src in final_sources:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<strong>{src["source"]}</strong>'
                                f'{(" · page " + str(src["page"])) if src.get("page") else ""}'
                                f'<br><span style="color:#666">{src["content"]}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

            except requests.exceptions.ConnectionError:
                full_answer = "Cannot connect to backend. Make sure FastAPI is running on port 8000."
                placeholder.error(full_answer)
            except Exception as e:
                full_answer = f"Error: {str(e)}"
                placeholder.error(full_answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "sources": final_sources,
        })
