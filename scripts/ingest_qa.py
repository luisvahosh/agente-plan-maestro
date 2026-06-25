#!/usr/bin/env python3
"""
Ingesta del material de entrenamiento (banco Q&A verificado) a Supabase.

Inyecta como CONOCIMIENTO las preguntas+respuestas verificadas por el equipo del PMDI,
para que el agente recupere respuestas curadas cuando un usuario pregunta algo parecido
(técnica de "semantic Q&A matching").

Fuentes (carpeta entrenamiento/):
  - PMDI_Banco_QA_Agente_Conversacional.docx  → 12 Q&A con nota de entrenamiento
  - preguntas_entrenamiento_pmdi.pdf          → 10 casos de articulación

Los chunks se insertan con chunk_id en rangos RESERVADOS para no colisionar con los
2420 chunks de los PDFs (0..~2419) y para ser idempotentes (re-ejecutable sin duplicar).

Uso:  python scripts/ingest_qa.py
"""

import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()

try:
    import fitz  # PyMuPDF
    import docx  # python-docx
    from openai import OpenAI
    from supabase import create_client, Client
except ImportError as e:
    print(f"❌ Falta dependencia: {e}. Instala: pip install pymupdf python-docx openai supabase")
    sys.exit(1)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

ENTR_DIR = Path("entrenamiento")
DOCX_QA = ENTR_DIR / "PMDI_Banco_QA_Agente_Conversacional.docx"
PDF_CASOS = ENTR_DIR / "preguntas_entrenamiento_pmdi.pdf"

SOURCE_QA = "Banco Q&A verificado PMDI"
SOURCE_CASOS = "Casos de articulación PMDI"
CHUNK_ID_QA = 1_000_000      # rango reservado banco Q&A
CHUNK_ID_CASOS = 1_100_000   # rango reservado casos

BATCH_SIZE = 10
MAX_RETRIES = 4


def validate_env():
    missing = [k for k, v in {
        "NVIDIA_API_KEY": NVIDIA_API_KEY,
        "SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_KEY": SUPABASE_KEY,
    }.items() if not v]
    if missing:
        print(f"❌ Faltan variables en .env: {', '.join(missing)}")
        sys.exit(1)


# ── Parseo del banco Q&A (Word) ───────────────────────────────────────────────

def parse_qa_docx(path: Path) -> list[dict]:
    """Extrae pares pregunta+respuesta del Word, excluyendo las notas de entrenamiento."""
    doc = docx.Document(path)
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    # Límite final: donde empieza la sección de recomendaciones
    end = len(paras)
    for i, t in enumerate(paras):
        if t.lower().startswith("recomendaciones para entrenar"):
            end = i
            break

    # Índices de inicio de cada bloque "Pregunta N"
    starts = [i for i in range(end) if re.match(r"^Pregunta\s+\d+$", paras[i])]
    starts.append(end)

    chunks = []
    for b in range(len(starts) - 1):
        block = paras[starts[b]: starts[b + 1]]

        # Localizar marcador "Respuesta"
        try:
            r_idx = next(i for i, t in enumerate(block) if t.lower() == "respuesta")
        except StopIteration:
            continue

        # Pregunta = líneas entre "Categoría:" y "Respuesta" (la última suele ser la pregunta)
        cat_idx = next((i for i, t in enumerate(block) if t.lower().startswith("categoría")), 0)
        pregunta = " ".join(block[cat_idx + 1: r_idx]).strip()

        # Respuesta = líneas tras "Respuesta", excluyendo la nota de entrenamiento (💡)
        respuesta_lines = [t for t in block[r_idx + 1:] if not t.startswith("💡")]
        respuesta = "\n".join(respuesta_lines).strip()

        if pregunta and respuesta:
            content = f"[{SOURCE_QA}]\nPregunta: {pregunta}\nRespuesta: {respuesta}"
            chunks.append({
                "chunk_id": CHUNK_ID_QA + len(chunks),
                "content": content.replace("\x00", ""),
                "source": SOURCE_QA,
            })
    return chunks


# ── Parseo de los casos (PDF) ─────────────────────────────────────────────────

def parse_casos_pdf(path: Path) -> list[dict]:
    """Extrae los 10 casos numerados del PDF (pregunta + respuesta juntos)."""
    d = fitz.open(path)
    text = "".join(p.get_text() for p in d)
    d.close()

    # Quitar todo lo anterior al primer "1. "
    m = re.search(r"(?m)^\s*1\.\s", text)
    if m:
        text = text[m.start():]

    # Quitar líneas que son solo número de página (footer)
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

    # Separar por inicio de ítem numerado "N. "
    pieces = re.split(r"(?m)^(?=\d{1,2}\.\s)", text)

    chunks = []
    for piece in pieces:
        piece = piece.strip()
        if not piece or not re.match(r"^\d{1,2}\.\s", piece):
            continue
        body = re.sub(r"^\d{1,2}\.\s*", "", piece).strip()
        content = f"[{SOURCE_CASOS}]\n{body}"
        chunks.append({
            "chunk_id": CHUNK_ID_CASOS + len(chunks),
            "content": content.replace("\x00", ""),
            "source": SOURCE_CASOS,
        })
    return chunks


# ── Embeddings + inserción ────────────────────────────────────────────────────

def generate_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.embeddings.create(
                input=texts, model=EMBED_MODEL,
                extra_body={"input_type": "passage", "truncate": "END"},
            )
            return [item.embedding for item in resp.data]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                print(f"  ⚠️  reintento {attempt+1}/{MAX_RETRIES} en {wait}s — {str(e)[:60]}")
                time.sleep(wait)
            else:
                raise


def main():
    validate_env()
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    print("🚀 Ingesta de material de entrenamiento (banco Q&A verificado)")

    chunks = []
    if DOCX_QA.exists():
        qa = parse_qa_docx(DOCX_QA)
        print(f"  ✓ {len(qa)} pares Q&A desde {DOCX_QA.name}")
        chunks += qa
    else:
        print(f"  ⚠️  No se encontró {DOCX_QA}")

    if PDF_CASOS.exists():
        casos = parse_casos_pdf(PDF_CASOS)
        print(f"  ✓ {len(casos)} casos desde {PDF_CASOS.name}")
        chunks += casos
    else:
        print(f"  ⚠️  No se encontró {PDF_CASOS}")

    if not chunks:
        print("❌ No hay nada que ingestar.")
        sys.exit(1)

    print(f"\n🔄 Vectorizando e insertando {len(chunks)} chunks…")
    inserted = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        embeddings = generate_embeddings(client, [c["content"] for c in batch])
        records = [
            {"chunk_id": c["chunk_id"], "content": c["content"],
             "source": c["source"], "embedding": emb}
            for c, emb in zip(batch, embeddings)
        ]
        resp = supabase.table("document_chunks").upsert(records, on_conflict="chunk_id").execute()
        inserted += len(resp.data)
        print(f"  lote {i // BATCH_SIZE + 1}: +{len(resp.data)}")

    print(f"\n✅ Listo: {inserted} chunks de entrenamiento en Supabase")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  cancelado")
        sys.exit(1)
