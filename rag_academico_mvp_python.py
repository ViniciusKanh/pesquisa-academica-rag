"""
RAG acadêmico — MVP em Python (local-first) com Re-ranking + Retry

Funcionalidades:
- Ingestão de PDFs (título + página), chunking com sobreposição
- Indexação vetorial persistente (ChromaDB)
- Consulta semântica + re-ranking (CrossEncoder) + geração com LLM (Ollama)
- Fallback extrativo e Retry "relaxado" quando o LLM disser "não encontrado"
- Citações [Título, p. N] em todas as respostas

Como usar (PowerShell):
1) python rag_academico_mvp_python.py clear
2) python rag_academico_mvp_python.py ingest .\meus_papers
3) python rag_academico_mvp_python.py ask "Sua pergunta" --k 6

Ambiente (recomendado):
$env:RAG_DB_PATH  = ".\rag_db"
$env:OLLAMA_HOST  = "http://localhost:11434"
$env:OLLAMA_MODEL = "llama3.1"
# Re-ranking:
$env:RERANK_ENABLED = "1"
# Debug (opcional):
$env:DEBUG_CONTEXTS = "1"
"""

from __future__ import annotations
import os
import sys
import glob
import argparse
import uuid
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

# ── Desativar telemetria/ruídos ANTES dos imports pesados
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TELEMETRY", "false")
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
from tqdm import tqdm

# Re-ranking (carregamento lazy dentro de ask)
_CE = None
CE_MODEL_NAME = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "0") == "1"
RERANK_CANDIDATES = int(os.environ.get("RERANK_CANDIDATES", "24"))  # nº docs para re-ranking
DEBUG_CONTEXTS = os.environ.get("DEBUG_CONTEXTS", "0") == "1"

# ── Tentar neutralizar posthog caso seja carregado por dependências
try:
    import posthog  # type: ignore

    def _no(*args, **kwargs):
        return None
    posthog.capture = _no  # type: ignore[attr-defined]
except Exception:
    pass

# =============================
# Configurações
# =============================
DB_PATH = os.environ.get("RAG_DB_PATH", "./rag_db")
COLLECTION_NAME = os.environ.get("RAG_COLLECTION", "papers")
EMB_MODEL_NAME = os.environ.get(
    "RAG_EMB_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
CHUNK_CHARS = int(os.environ.get("RAG_CHUNK_CHARS", 1200))   # ~200–300 tokens
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", 200))
TOP_K = int(os.environ.get("RAG_TOP_K", 6))

# LLM local via Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", 700))  # um pouco maior p/ síntese

SYSTEM_PROMPT_STRICT = (
    "Você é um assistente acadêmico. Responda SOMENTE com base nos trechos fornecidos entre <<<TRECHOS>>>.\n"
    "Considere 'seleção de atributos' apenas métodos FILTER (mRMR/ReliefF/MI/chi2/ANOVA/JMI/CIFE), WRAPPER (RFE/SFS/SBS/SFFS) e EMBEDDED (L1/LASSO/Elastic Net/árvores/ganho de XGBoost/permutation importance/Boruta/SHAP).\n"
    "NÃO confunda com reamostragem (SMOTE/under/over-sampling) nem redução de dimensionalidade (PCA/t-SNE/autoencoders). NÃO cite esses termos como seleção de atributos.\n"
    "Se a resposta completa não estiver presente, apresente uma resposta PARCIAL com limitações.\n"
    "Seja preciso, conciso e cite as fontes no final no formato [Título, p. N].\n"
)

SYSTEM_PROMPT_RELAXED = (
    "Você é um assistente acadêmico. Responda com base nos trechos fornecidos entre <<<TRECHOS>>>.\n"
    "Mantenha a taxonomia: métodos de seleção de atributos = FILTER/WRAPPER/EMBEDDED; NÃO cite reamostragem (SMOTE etc.) nem redução de dimensionalidade (PCA/t-SNE) como seleção.\n"
    "Se os trechos forem insuficientes, forneça um resumo do que há e indique lacunas. Cite as fontes [Título, p. N].\n"
)


# =============================
# Utilitários
# =============================

def split_text(text: str, max_chars: int, overlap: int) -> List[str]:
    """Chunking simples por caracteres, com sobreposição."""
    t = (text or "").strip()
    if not t:
        return []
    chunks: List[str] = []
    start, n = 0, len(t)
    while start < n:
        end = min(start + max_chars, n)
        chunk = t[start:end]
        # opcional: descartar trechos muito curtos (ruído)
        if len(chunk.strip()) >= 60:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


@dataclass
class DocChunk:
    text: str
    title: str
    page: int
    source: str  # caminho do arquivo


def extract_pdf_chunks(pdf_path: str, max_chars: int, overlap: int) -> List[DocChunk]:
    """Extrai texto por página com PyMuPDF; ignora páginas sem texto."""
    doc = fitz.open(pdf_path)
    title = os.path.splitext(os.path.basename(pdf_path))[0]
    chunks: List[DocChunk] = []
    try:
        for i in range(len(doc)):
            page = doc[i]
            raw = page.get_text("text") or ""
            txt = raw.strip()
            if not txt:
                continue
            for ch in split_text(txt, max_chars, overlap):
                chunks.append(DocChunk(text=ch, title=title, page=i + 1, source=pdf_path))
    finally:
        doc.close()
    return chunks


# =============================
# Vetorização & Base Vetorial
# =============================

class Embedder:
    def __init__(self, model_name: str = EMB_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )


class VectorStore:
    def __init__(self, path: str = DB_PATH, collection: str = COLLECTION_NAME):
        self.client = chromadb.PersistentClient(
            path=path,
            settings=Settings(
                allow_reset=False,
                anonymized_telemetry=False,  # força OFF
            ),
        )
        self.collection = self.client.get_or_create_collection(name=collection)

    def clear(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(name=self.collection.name)

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: np.ndarray | List[List[float]] | None = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        if embeddings is not None:
            payload["embeddings"] = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        self.collection.add(**payload)

    def query(
        self,
        query_texts: List[str] | None = None,
        query_embeddings: List[List[float]] | None = None,
        n_results: int = 6,
    ):
        return self.collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )


# =============================
# LLM Backend (Ollama) & Fallback
# =============================

def _ollama_chat(system_prompt: str, user_prompt: str) -> str:
    """Chat via Ollama (/api/chat). Em falha, retorna string vazia."""
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"num_predict": LLM_MAX_TOKENS},
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("message", {}) or {}).get("content", "") or ""
    except Exception:
        return ""


def _build_prompt(question: str, contexts: List[str]) -> str:
    header = "<<<TRECHOS>>>\n"
    joined = "\n\n---\n\n".join(contexts)
    tail = "\n<<<FIM TRECHOS>>>\n\nPergunta: " + question.strip()
    return header + joined + tail


def _extractive_summary(contexts: List[str], max_items: int = 3, max_chars_per_item: int = 350) -> str:
    """Resumo extrativo curto dos trechos (sem LLM)."""
    bullets: List[str] = []
    for ctx in contexts[:max_items]:
        lines = ctx.splitlines()
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ctx  # pula [Fonte: ...]
        body = body.replace("\r", " ").replace("\n", " ").strip()
        snippet = body[:max_chars_per_item] + ("…" if len(body) > max_chars_per_item else "")
        bullets.append(f"- {snippet}")
    return "Resumo dos trechos recuperados:\n" + "\n".join(bullets)


def _answer_with_retry(question: str, contexts: List[str], citations: List[Tuple[str, int]]) -> str:
    """1ª tentativa: prompt estrito; se vier 'não encontrado', tenta novamente com prompt relaxado."""
    user_prompt = _build_prompt(question, contexts)

    out1 = _ollama_chat(SYSTEM_PROMPT_STRICT, user_prompt).strip()
    not_found = "não encontrado nos artigos fornecidos" in out1.lower()

    if (out1 and not not_found):
        answer = out1
    else:
        # Retry com prompt relaxado
        out2 = _ollama_chat(SYSTEM_PROMPT_RELAXED, user_prompt).strip()
        if out2:
            answer = out2
        else:
            # Fallback extrativo
            answer = _extractive_summary(contexts)

    # Citações únicas [Título, p. N]
    uniq: List[Tuple[str, int]] = []
    seen = set()
    for (title, page) in citations:
        key = (title, page)
        if key not in seen:
            seen.add(key)
            uniq.append(key)
    cites = " ".join([f"[{t}, p. {p}]" for (t, p) in uniq])
    if cites:
        answer = answer.rstrip() + f"\n\nReferências: {cites}"
    return answer


# =============================
# Pipeline principal
# =============================

def ingest(folder: str) -> None:
    """Extrai, deduplica, embeda e indexa chunks dos PDFs."""
    if os.path.isdir(folder):
        paths = sorted(glob.glob(os.path.join(folder, "**", "*.pdf"), recursive=True))
    elif folder.lower().endswith(".pdf"):
        paths = [folder]
    else:
        print("[ERRO] Informe uma pasta ou arquivo .pdf")
        sys.exit(1)

    if not paths:
        print("[AVISO] Nenhum PDF encontrado.")
        return

    print(f"Encontrados {len(paths)} PDF(s). Extraindo...")
    embedder = Embedder(EMB_MODEL_NAME)
    vs = VectorStore(DB_PATH, COLLECTION_NAME)

    seen_hashes: set[str] = set()
    total_chunks = 0

    for pdf in tqdm(paths, desc="Processando PDFs"):
        chunks = extract_pdf_chunks(pdf, CHUNK_CHARS, CHUNK_OVERLAP)
        if not chunks:
            continue

        docs_all = [c.text for c in chunks]
        metas_all = [{"title": c.title, "page": c.page, "source": c.source} for c in chunks]

        # Deduplicação por conteúdo (hash blake2s)
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        for d, m in zip(docs_all, metas_all):
            h = hashlib.blake2s(d.encode("utf-8"), digest_size=16).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            docs.append(d)
            metas.append(m)

        if not docs:
            continue

        ids = [uuid.uuid4().hex for _ in docs]
        embs = embedder.encode(docs)
        vs.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
        total_chunks += len(docs)

    print(f"OK. {total_chunks} chunk(s) indexados em {DB_PATH} / coleção '{COLLECTION_NAME}'.")


def _maybe_rerank(query: str, docs: List[str], metas: List[Dict[str, Any]], top_k: int) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Re-ranking com CrossEncoder (se habilitado)."""
    global _CE
    if not RERANK_ENABLED or not docs:
        return docs[:top_k], metas[:top_k]
    try:
        if _CE is None:
            from sentence_transformers import CrossEncoder  # import lazy
            _CE = CrossEncoder(CE_MODEL_NAME)  # baixa na 1ª vez
        # pegar candidatos
        cand_n = min(len(docs), max(top_k, RERANK_CANDIDATES))
        pairs = [(query, d) for d in docs[:cand_n]]
        scores = _CE.predict(pairs)  # maior = mais relevante
        order = np.argsort(-np.array(scores))[:top_k]
        docs_r = [docs[i] for i in order]
        metas_r = [metas[i] for i in order]
        return docs_r, metas_r
    except Exception as e:
        # se der problema (modelo não baixado / sem internet), retorna sem re-ranking
        return docs[:top_k], metas[:top_k]


def ask(query: str, top_k: int = TOP_K) -> str:
    """Recupera trechos (com re-ranking opcional) e responde (LLM ou extrativo), com citações."""
    vs = VectorStore(DB_PATH, COLLECTION_NAME)
    embedder = Embedder(EMB_MODEL_NAME)

    # Recupera candidatos (mais do que top_k para re-ranking)
    initial_k = max(top_k, RERANK_CANDIDATES) if RERANK_ENABLED else top_k
    q_emb = embedder.encode([query])[0]
    res = vs.query(query_embeddings=[q_emb.tolist()], n_results=initial_k)

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    if not docs:
        return "Nenhum trecho relevante encontrado no índice. Ingestione PDFs primeiro."

    # Re-ranking (opcional)
    docs, metas = _maybe_rerank(query, docs, metas, top_k)

    # Monta contextos e citações
    contexts: List[str] = []
    citations: List[Tuple[str, int]] = []
    for d, m in zip(docs, metas):
        title = m.get("title", "(sem título)")
        page = m.get("page", -1)
        d = (d or "").strip()
        if not d:
            continue
        contexts.append(f"[Fonte: {title}, p. {page}]\n{d}")
        citations.append((title, page))

    if not contexts:
        return "Nenhum trecho relevante encontrado no índice. Ingestione PDFs primeiro."

    if DEBUG_CONTEXTS:
        print("\n[DEBUG] Contextos selecionados:")
        for i, ctx in enumerate(contexts, 1):
            head = ctx.splitlines()[0]
            print(f"{i:02d}. {head}")

    return _answer_with_retry(query, contexts, citations)


def clear_index() -> None:
    vs = VectorStore(DB_PATH, COLLECTION_NAME)
    vs.clear()
    print("Índice limpo com sucesso.")


# =============================
# CLI
# =============================

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG acadêmico — MVP (local-first) c/ re-ranking")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Ingerir PDFs para o índice")
    p_ing.add_argument("path", type=str, help="Pasta com PDFs ou arquivo .pdf")

    p_ask = sub.add_parser("ask", help="Fazer uma pergunta baseada no índice")
    p_ask.add_argument("question", type=str, help="Pergunta em linguagem natural")
    p_ask.add_argument("--k", type=int, default=TOP_K, help="Número de trechos recuperados")

    sub.add_parser("clear", help="Limpar o índice vetorial")

    args = parser.parse_args()
    if args.cmd == "ingest":
        ingest(args.path)
    elif args.cmd == "ask":
        print(ask(args.question, top_k=args.k))
    elif args.cmd == "clear":
        clear_index()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
