# server.py
import os
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

# Importa as funções do seu RAG (mesmo diretório)
from rag_academico_mvp_python import (
    ingest as rag_ingest,
    ask as rag_ask,
    clear_index as rag_clear,
)

# Pastas padrão
DATA_DIR = os.environ.get("DATA_DIR", "./data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
RAG_DB_PATH = os.environ.get("RAG_DB_PATH", "./rag_db")

# Garante diretórios
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RAG_DB_PATH, exist_ok=True)

app = FastAPI(title="RAG Acadêmico API", version="1.0.0")

# CORS liberado para facilitar frontend local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # em produção: restrinja
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    k: int = 6

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/clear")
async def clear_index():
    await run_in_threadpool(rag_clear)
    return {"status": "cleared"}

@app.post("/ingest_files")
async def ingest_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")
    saved = []
    for f in files:
        # valida extensão
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Apenas PDF permitido: {f.filename}")
        dest = os.path.join(UPLOAD_DIR, f.filename)
        content = await f.read()
        with open(dest, "wb") as w:
            w.write(content)
        saved.append(dest)

    # ingere a pasta de uploads (bloqueante → roda em threadpool)
    await run_in_threadpool(rag_ingest, UPLOAD_DIR)

    return {"status": "ok", "files": [os.path.basename(p) for p in saved]}

@app.post("/ask")
async def ask(req: AskRequest):
    # rag_ask é CPU-bound; rodar em threadpool evita travar o loop
    answer = await run_in_threadpool(rag_ask, req.question, req.k)
    return {"answer": answer}
