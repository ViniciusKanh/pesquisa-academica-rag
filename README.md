
# Pesquisa Academica RAG

RAG (Retrieval-Augmented Generation) para **busca, leitura e sumarização de artigos acadêmicos em PT-BR**, com pipeline de **ingestão**, **indexação FAISS**, **embeddings Sentence-Transformers** e **API FastAPI**. Suporta LLM local (Transformers/Ollama) ou serviços (OpenAI/HF Inference).

> **Objetivo**: permitir que qualquer pessoa carregue PDFs/textos acadêmicos e faça perguntas naturais, recebendo respostas citadas e passagens de suporte.

---

## ✨ Funcionalidades

* **Ingestão** de PDFs/TXT/MD/HTML com chunking configurável
* **Embeddings** multilíngues (recomendado: `intfloat/multilingual-e5-base`)
* **Indexação vetorial** com **FAISS** (CPU por padrão)
* **Busca híbrida** (BM25 opcional) + Semântica (kNN / MMR)
* **RAG**: recuperação de trechos + geração com LLM
* **API REST** (FastAPI) com **/docs (Swagger)**
* **CLI** para ingestão e queries locais
* **Docker** pronto (CPU)
* **Citações**: trechos e metadados da fonte retornados junto à resposta
* **Configuração via `.env`** (modelos, caminhos, top\_k, temperatura…)

---

## 🏗️ Arquitetura & Fluxo

```
                +--------------------+
                |   Usuário/Cliente  |
                |  (cURL/Swagger/CLI)|
                +----------+---------+
                           |
                           v
                     FastAPI /query
                           |
                           v
   +---------------- Recuperador ----------------+
   |  FAISS Index  |  (BM25 opc.) |  Filtros     |
   |  embeddings   |  (rank/merge)|  metadados   |
   +--------+------+--------------+--------------+
            | top_k
            v
       Passagens     +------------------------------+
   +---------------->|   Gerador (LLM)              |
   |                 |  (Transformers/Ollama/OpenAI)|
   |                 +------------------------------+
   |                              |
   |<------------- Resposta + citações -------------+
```

**Pastas sugeridas**

```
app/
  main.py               # API FastAPI
  rag/
    pipeline.py         # Orquestra RAG
    retriever.py        # Busca FAISS (+ BM25 opcional)
    embedder.py         # Carrega modelo de embeddings
    generator.py        # Abstrai backends LLM
    utils.py            # Parse, chunking, etc.
configs/
  settings.py           # Pydantic/BaseSettings (lê .env)
data/
  docs/                 # Documentos brutos (não versionado)
  index/                # Índices FAISS persistidos
scripts/
  ingest.py             # CLI: ingestão e indexação
  query.py              # CLI: pergunta rápida via terminal
tests/
  ...
```

---

## ✅ Requisitos

* **Python 3.11+**
* **pip/venv** (ou conda/poetry)
* **CPU** basta; **GPU** opcional (PyTorch com CUDA)
* **faiss-cpu** (ou `faiss-gpu` se desejar)
* Acesso à internet na primeira execução para baixar modelos (se usar Transformers) — ou use **Ollama**/serviço externo.

---

## ⚙️ Instalação

```bash
# 1) Clonar o repo
git clone https://github.com/SEU-USUARIO/pesquisa-academica-rag.git
cd pesquisa-academica-rag

# 2) Ambiente virtual
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 3) Dependências
pip install --upgrade pip
pip install -r requirements.txt
# Obs.: em alguns ambientes Linux é necessário:
# pip install faiss-cpu --no-cache-dir
```

### `requirements.txt` (sugestão)

```txt
fastapi>=0.111
uvicorn[standard]>=0.30
python-dotenv>=1.0
pydantic>=2.7
sentence-transformers>=2.7
faiss-cpu>=1.8
transformers>=4.43
accelerate>=0.33
# Instale o torch adequado ao seu SO/GPU:
torch>=2.3
pypdf>=4.2
tqdm>=4.66
httpx>=0.27
rich>=13.7
typer>=0.12
orjson>=3.10
```

> **Torch**: caso tenha GPU NVIDIA, instale a versão com CUDA (veja instruções no site do PyTorch).

---

## 🔧 Configuração

Crie um `.env` na raiz (baseado no exemplo abaixo):

```env
# ===== Embeddings =====
EMBEDDINGS_MODEL=intfloat/multilingual-e5-base

# ===== Indexação =====
DOCS_DIR=./data/docs
INDEX_DIR=./data/index
CHUNK_SIZE=512
CHUNK_OVERLAP=64
USE_BM25=false

# ===== Recuperação =====
TOP_K=5
MMR=false
MMR_LAMBDA=0.5

# ===== LLM (escolha um backend) =====
# transformers | openai | ollama | hf_inference
LLM_BACKEND=transformers

# Transformers (local)
LLM_MODEL=google/gemma-2-2b-it
MAX_TOKENS=512
TEMPERATURE=0.2

# OpenAI
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b-instruct

# HF Inference API
HF_API_KEY=
HF_INFERENCE_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# ===== API =====
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info
```

> **Dicas de modelo**
>
> * **Embeddings**: `intfloat/multilingual-e5-base` (excelente para PT-BR).
> * **LLM leve (CPU)**: `google/gemma-2-2b-it` (Transformers) ou `llama3:8b-instruct` via Ollama.
> * **Serviço**: `gpt-4o-mini` (OpenAI) ou `Mistral-7B-Instruct` (HF Inference).

---

## 🧾 Ingestão de Documentos

Coloque seus arquivos em `./data/docs` (ou passe `--path`).

Formatos suportados inicialmente: **.pdf, .txt, .md, .html**

```bash
# ingestão + indexação
python scripts/ingest.py --path ./data/docs \
  --chunk-size 512 --chunk-overlap 64 \
  --recreate
```

**Parâmetros comuns**

* `--path`: pasta com documentos
* `--recreate`: recria o índice do zero
* `--append`: adiciona ao índice existente
* `--chunk-size`, `--chunk-overlap`: controle de segmentação
* `--pattern "*.pdf"`: filtra extensão

> Após a ingestão, o índice FAISS é salvo em `./data/index`.

---

## 🚀 Executando a API

```bash
# com .env configurado
uvicorn app.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000} --reload
# Abra: http://localhost:8000/docs
```

### Endpoints principais

* `GET /health` – saúde do serviço
* `POST /query` – pergunta com RAG
* `POST /rerank` (opcional) – reordena passagens recuperadas
* `POST /ingest` (opcional) – ingestão via API (para automações)

#### `POST /query` – Exemplo de payload

```json
{
  "question": "Quais são os principais métodos para detecção de tópicos em português?",
  "top_k": 5,
  "mmr": false,
  "filters": {
    "source": ["meuartigo.pdf"]
  }
}
```

#### Resposta (exemplo)

```json
{
  "answer": "Os métodos mais comuns incluem LDA, NMF e BERTopic...",
  "citations": [
    {
      "doc_id": "meuartigo.pdf",
      "page": 7,
      "score": 0.82,
      "text": "O BERTopic combina embeddings e clustering HDBSCAN..."
    }
  ],
  "used_model": "google/gemma-2-2b-it",
  "top_k": 5,
  "timing_ms": 842
}
```

---

## 🧪 Exemplos de uso

### cURL

```bash
curl -s http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Resuma as contribuições do artigo X","top_k":5}' | jq
```

### Python

```python
import httpx, json
payload = {"question": "Quais datasets foram usados no estudo Y?", "top_k": 5}
r = httpx.post("http://localhost:8000/query", json=payload, timeout=120)
print(json.dumps(r.json(), ensure_ascii=False, indent=2))
```

### CLI rápida (sem API)

```bash
python scripts/query.py --q "Explique o método proposto no paper Z" --top-k 5
```

---

## 🐳 Docker

### Build & Run (CPU)

```bash
docker build -t pesquisa-academica-rag .
docker run --rm -it -p 8000:8000 \
  --env-file .env \
  -v $PWD/data:/app/data \
  pesquisa-academica-rag
# http://localhost:8000/docs
```

> O volume `-v $PWD/data:/app/data` garante persistência do índice FAISS.

### Compose (opcional)

```yaml
# docker-compose.yml
services:
  rag:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
```

```bash
docker compose up --build
```

---

## ⚡ Performance & Qualidade

* **Embeddings**: E5 é robusto em PT-BR; para mais precisão tente `multilingual-e5-large` (custa mais RAM).
* **Chunking**: comece com `512/64`. Conteúdo muito técnico pode melhorar com `800-1000` tokens.
* **top\_k**: 5–8 é um bom range; ative **MMR** em coleções redundantes.
* **Reranking**: adicione um reranker (ex.: `cross-encoder/ms-marco-MiniLM-L-6-v2`) se a precisão de recuperação for crítica.
* **Cache**: ative cache de embeddings/geração (ex.: disco) para acelerar re-execuções.
* **GPU**: se disponível, habilite `device_map="auto"` no `generator.py`/`embedder.py`.

---

## 🩺 Troubleshooting

* **FAISS não instala**: use `faiss-cpu` (ou `conda install -c pytorch faiss-cpu`).
* **Torch (GPU)**: instale a variante correta (CUDA 11/12) no site do PyTorch.
* **Memória insuficiente**: use um LLM menor (`gemma-2-2b-it`/Ollama) ou reduza `MAX_TOKENS`.
* **PDFs com texto “colado”**: alguns PDFs têm OCR ruim; use uma etapa de OCR (Tesseract) antes de ingerir.
* **Resultados ruins**: aumente `CHUNK_SIZE`, ajuste `TOP_K`, ative `MMR`, e revise a limpeza dos textos.

---

## 🔒 Segurança & Ética

* Não suba PDFs proprietários sem permissão.
* Respeite direitos autorais e **cite fontes**.
* O modelo pode alucinar; use as **citações** para verificar passagens.

---

## 🧭 Roadmap (sugestão)

* [ ] UI web simples (chat + upload)
* [ ] Reranking com Cross-Encoder
* [ ] Suporte nativo a **OCRTesseract** na ingestão
* [ ] Exportar respostas + citações em **BibTeX**
* [ ] Avaliação RAG automática (Precision\@k, MRR, F1)
* [ ] Multi-índice + filtros avançados por metadados

---

## 🧹 Scripts úteis

```bash
# Recriar índice do zero
python scripts/ingest.py --path ./data/docs --recreate

# Adicionar novos docs
python scripts/ingest.py --path ./data/docs --append

# Perguntar por CLI
python scripts/query.py --q "Qual é a contribuição central do artigo A?"
```

---

## 🧪 Testes

* Testes unitários de utilitários e do retriever (mocks) em `tests/`.
* Sugestão (pytest):

```bash
pip install pytest
pytest -q
```

---

## 📄 Licença

**MIT** — sinta-se livre para usar e melhorar. Inclua as citações adequadas ao compartilhar resultados.

---

## 🤝 Contribuindo

Contribuições são bem-vindas!

1. Faça um fork
2. Crie uma branch: `feat/minha-ideia`
3. Commit: `feat: descrição`
4. PR com descrição clara (screenshots e exemplos ajudam!)

---

## 🙏 Agradecimentos

* [Sentence-Transformers](https://www.sbert.net/)
* [FAISS](https://faiss.ai/)
* [Transformers](https://huggingface.co/docs/transformers/)
* Comunidade RAG/open-source 🚀

---

## 📎 Anexos (exemplos rápidos de código)

### `scripts/ingest.py` (esqueleto mínimo)

```python
# scripts/ingest.py
import argparse, os
from app.rag.embedder import Embeddings
from app.rag.retriever import FaissIndex
from app.rag.utils import load_and_chunk_all

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--chunk-size", type=int, default=int(os.getenv("CHUNK_SIZE", 512)))
    p.add_argument("--chunk-overlap", type=int, default=int(os.getenv("CHUNK_OVERLAP", 64)))
    m = p.add_mutually_exclusive_group()
    m.add_argument("--recreate", action="store_true")
    m.add_argument("--append", action="store_true")
    args = p.parse_args()

    texts, metas = load_and_chunk_all(args.path, args.chunk_size, args.chunk_overlap)
    emb = Embeddings()
    idx = FaissIndex(recreate=args.recreate)
    idx.add(texts, metas, emb)
    idx.save()
    print(f"Index pronto em {os.getenv('INDEX_DIR','./data/index')}")

if __name__ == "__main__":
    main()
```

### `scripts/query.py` (esqueleto mínimo)

```python
# scripts/query.py
import argparse
from app.rag.pipeline import RagPipeline

p = argparse.ArgumentParser()
p.add_argument("--q", required=True)
p.add_argument("--top-k", type=int, default=5)
args = p.parse_args()

rag = RagPipeline()
out = rag.answer(args.q, top_k=args.top_k)
print(out["answer"])
for c in out["citations"]:
    print(f"- {c['doc_id']} (score={c['score']:.2f}): {c['text'][:120]}...")
```

