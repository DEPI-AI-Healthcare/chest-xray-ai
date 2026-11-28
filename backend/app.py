# InnoTech AI Backend - LEAN version for Railway (chat only)
#
# No TensorFlow / torch / torchxrayvision here: they exceed Railway's RAM and
# the server was being OOM-killed. X-ray analysis (/predict) stays in Colab
# (app_full.py). This service handles login + /chat (Groq LLM + optional RAG).
#
# Railway Variables:
#   GROQ_API_KEY     (required)
#   LLM_MODEL        (optional, default openai/gpt-oss-120b)
#   MAX_TOKENS       (optional, default 1500)
#   FAISS_INDEX_DIR  (optional, default ./faiss_index) - folder with index.faiss + index.pkl
#   PREDICT_URL      (optional) - public URL of the Colab/ngrok X-ray server; if set,
#                    /predict forwards the uploaded image there.

import os
import re
import uuid
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="InnoTech AI Backend", version="3.3.0-lean")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./faiss_index")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))
PREDICT_URL = os.getenv("PREDICT_URL", "").rstrip("/")

# ---------------------------------------------------------------------------
# Optional lightweight RAG (ONNX embeddings, same MiniLM model as the index)
# ---------------------------------------------------------------------------
retriever = None
try:
    if os.path.isdir(FAISS_INDEX_DIR):
        from langchain_core.embeddings import Embeddings
        from langchain_community.vectorstores import FAISS
        from fastembed import TextEmbedding

        class FastEmbedEmbeddings(Embeddings):
            def __init__(self, model_name: str):
                self.model = TextEmbedding(model_name=model_name)

            def embed_documents(self, texts: List[str]):
                return [v.tolist() for v in self.model.embed(texts)]

            def embed_query(self, text: str):
                return next(iter(self.model.embed([text]))).tolist()

        _emb = FastEmbedEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
        _store = FAISS.load_local(FAISS_INDEX_DIR, _emb, allow_dangerous_deserialization=True)
        retriever = _store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        print("[startup] RAG index loaded.")
    else:
        print(f"[startup] No FAISS index at {FAISS_INDEX_DIR}; /chat answers without report context.")
except Exception as e:
    retriever = None
    print(f"[startup] RAG disabled: {type(e).__name__}: {e}")

llm_client = None
if GROQ_API_KEY:
    llm_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
else:
    print("[startup] GROQ_API_KEY is not set; /chat will return 503.")

SYSTEM_PROMPT = """You are an AI clinical assistant for chest radiology and lung screening (such as LDCT eligibility and general pulmonary health).
Rules:
- If the user asks about specific radiology findings, rely strictly on the provided report context below.
- If the user asks general educational questions about lung screening (e.g., LDCT criteria, symptoms, follow-up guidelines), provide accurate, concise clinical guidance without prescribing medications or dosages.
- Never issue a definitive medical diagnosis; always remind that findings require physician review.
- Respond in the same language as the user's question (English or Arabic).
- Answer in plain text without Markdown formatting (no bold **, no # headings, no Markdown tables). Use short paragraphs or simple numbered lines.

Retrieved Radiology Context:
{context}"""


_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def strip_markdown(text: str) -> str:
    """Turn Markdown into clean plain text for the Android app."""
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_row = line.strip().startswith("|") and line.strip().endswith("|")
        if is_row:
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if _TABLE_SEP.match(nxt):        # header row of a table -> skip it and its separator
                i += 2
                continue
            if _TABLE_SEP.match(line):
                i += 1
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            cells = [c for c in cells if c]
            if len(cells) == 2:
                out.append(f"- {cells[0]}: {cells[1]}")
            elif cells:
                out.append("- " + " | ".join(cells))
            i += 1
            continue
        out.append(line)
        i += 1

    t = "\n".join(out)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.MULTILINE)   # headings
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)                       # bold
    t = re.sub(r"__(.+?)__", r"\1", t)
    t = re.sub(r"^\s*\*\s+", "- ", t, flags=re.MULTILINE)           # * bullets -> -
    t = re.sub(r"`([^`]+)`", r"\1", t)                              # inline code
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def generate_rag_answer(question: str):
    if llm_client is None:
        raise RuntimeError(
            "LLM is not configured: set GROQ_API_KEY in Railway -> Variables and redeploy."
        )
    docs = retriever.invoke(question) if retriever is not None else []
    context = "\n\n".join(d.page_content for d in docs) if docs else "No specific matching report found."

    kwargs = {}
    if "gpt-oss" in LLM_MODEL:
        # reasoning model: keep thinking short so it doesn't eat the token budget
        kwargs["extra_body"] = {"reasoning_effort": "low"}

    completion = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0,
        **kwargs,
    )
    answer = strip_markdown((completion.choices[0].message.content or "").strip())
    if not answer:
        raise RuntimeError("The model returned an empty answer; try again or raise MAX_TOKENS.")
    sources = [{"uid": d.metadata.get("uid", "N/A"), "snippet": d.page_content} for d in docs]
    return answer, sources


chat_history_store = []


@app.get("/")
def root():
    return {"status": "online", "service": "InnoTech AI Backend", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_configured": llm_client is not None,
        "rag_loaded": retriever is not None,
    }


class LoginData(BaseModel):
    email: str
    password: str


def handle_login(data: LoginData):
    if data.email == "doctor@hospital.com" and data.password == "123456":
        return {
            "status": "success",
            "access_token": f"innotech-jwt-{uuid.uuid4().hex}",
            "token_type": "bearer",
            "user": {"email": data.email, "role": "physician"},
        }
    raise HTTPException(status_code=401, detail="Invalid email or password")


@app.post("/login")
def login(data: LoginData):
    return handle_login(data)


@app.post("/api/v1/auth/login")
def api_login(data: LoginData):
    return handle_login(data)


@app.post("/predict")
def predict_xray(file: UploadFile = File(...)):
    if not PREDICT_URL:
        raise HTTPException(
            status_code=503,
            detail="X-ray analysis is not hosted on this server. Set PREDICT_URL to the Colab/ngrok server.",
        )
    import httpx
    try:
        r = httpx.post(
            f"{PREDICT_URL}/predict",
            files={"file": (file.filename or "xray.png", file.file.read(), file.content_type or "image/png")},
            timeout=120,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"X-ray server unreachable: {type(e).__name__}: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text[:500])
    return r.json()


class ChatRequest(BaseModel):
    question: Optional[str] = None
    message: Optional[str] = None


def process_chat(data: ChatRequest):
    user_query = (data.message or data.question or "").strip()
    if not user_query:
        raise HTTPException(status_code=422, detail="Either 'message' or 'question' is required.")
    try:
        answer, sources = generate_rag_answer(user_query)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[/chat] failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=503, detail=f"Chat backend error: {type(e).__name__}: {e}")

    chat_history_store.append({"sender": "doctor", "message": user_query})
    chat_history_store.append({"sender": "ai", "message": answer})
    return {
        "status": "success",
        "question": user_query,
        "message": user_query,
        "answer": answer,
        "response": answer,
        "sources": sources,
    }


@app.post("/chat")
def medical_chat(data: ChatRequest):
    return process_chat(data)


@app.post("/api/v1/ai/chat/messages")
def mobile_medical_chat(data: ChatRequest):
    return process_chat(data)


# trigger railway build
@app.get("/chat/history")
def get_chat_history():
    return {"status": "success", "history": chat_history_store}
