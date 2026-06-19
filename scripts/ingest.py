#!/usr/bin/env python3
"""
Ingesta del Plan Maestro Medellín Inteligente.

Fuente de verdad: PDFs en la carpeta Documentos/
Proceso:
  1. Extrae texto de cada PDF con PyMuPDF
  2. Divide en chunks con solapamiento
  3. Genera embeddings con NVIDIA
  4. Inserta/actualiza en Supabase pgvector

Uso:
  python scripts/ingest.py                   # usa PDFs en ./Documentos
  python scripts/ingest.py --docs /ruta/pdfs # carpeta alternativa
  python scripts/ingest.py --from-json       # usa chunks_plan_maestro.json (fallback)
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# Forzar UTF-8 en consola Windows (evita UnicodeEncodeError con emojis)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()

# ── Dependencias ──────────────────────────────────────────────────────────────
try:
    import fitz  # PyMuPDF
except ImportError:
    print("❌ Instala PyMuPDF: pip install pymupdf")
    sys.exit(1)

try:
    from openai import OpenAI
    from supabase import create_client, Client
except ImportError:
    print("❌ Instala dependencias: pip install openai supabase")
    sys.exit(1)

# ── Configuración ─────────────────────────────────────────────────────────────
NVIDIA_API_KEY  = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
SUPABASE_URL    = os.getenv("SUPABASE_URL")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY")

DOCS_DIR        = Path("Documentos")
CHUNKS_JSON     = Path("chunks_plan_maestro.json")
CHUNK_SIZE      = 1000   # caracteres
CHUNK_OVERLAP   = 200
BATCH_SIZE      = 10
MAX_RETRIES     = 4
RETRY_BASE_SECS = 2


def validate_env():
    missing = []
    if not NVIDIA_API_KEY:
        missing.append("NVIDIA_API_KEY")
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_KEY")
    if missing:
        print(f"❌ Faltan variables en .env: {', '.join(missing)}")
        sys.exit(1)


# ── Extracción de texto ───────────────────────────────────────────────────────

def extract_text(pdf_path: Path) -> str:
    """Extrae todo el texto de un PDF."""
    doc = fitz.open(pdf_path)
    pages = [doc.load_page(i).get_text() for i in range(len(doc))]
    doc.close()
    return "\n".join(pages)


def split_text(text: str, source: str, chunk_id_start: int) -> list[dict]:
    """Divide texto en chunks con solapamiento."""
    chunks = []
    start = 0
    chunk_id = chunk_id_start

    while start < len(text):
        end = start + CHUNK_SIZE
        snippet = text[start:end]

        # Limpiar caracteres nulos que PostgreSQL no acepta ()
        snippet = snippet.replace("\x00", "")

        # Evitar chunks de solo espacios/saltos de línea
        if snippet.strip():
            chunks.append({
                "chunk_id": chunk_id,
                "content": f"[{source}]\n{snippet.strip()}",
                "source": source,
            })
            chunk_id += 1

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def load_chunks_from_pdfs(docs_dir: Path) -> list[dict]:
    """Lee todos los PDFs de la carpeta y genera la lista de chunks."""
    pdf_files = sorted(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No se encontraron PDFs en {docs_dir}/")
        sys.exit(1)

    print(f"\n📂 {len(pdf_files)} PDFs encontrados en {docs_dir}/")
    all_chunks = []

    for pdf_path in pdf_files:
        print(f"  → {pdf_path.name}", end=" ")
        try:
            text = extract_text(pdf_path)
            chunks = split_text(text, pdf_path.name, len(all_chunks))
            all_chunks.extend(chunks)
            print(f"({len(chunks)} chunks)")
        except Exception as e:
            print(f"⚠️  Error: {e}")

    print(f"\n✓ Total chunks generados: {len(all_chunks)}")
    return all_chunks


def load_chunks_from_json(json_path: Path) -> list[dict]:
    """Carga chunks desde el JSON pre-generado (fallback)."""
    if not json_path.exists():
        print(f"❌ No se encontró {json_path}")
        sys.exit(1)
    with open(json_path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"✓ {len(chunks)} chunks cargados desde {json_path}")
    return chunks


# ── Embeddings ────────────────────────────────────────────────────────────────

def generate_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Genera embeddings con reintentos exponenciales.

    input_type='passage' es obligatorio en NVIDIA para contenido indexado.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                input=texts,
                model=EMBED_MODEL,
                extra_body={"input_type": "passage", "truncate": "END"},
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE_SECS ** (attempt + 1)
                print(f"\n  ⚠️  Reintento {attempt + 1}/{MAX_RETRIES} en {wait}s — {str(e)[:60]}")
                time.sleep(wait)
            else:
                raise


# ── Inserción en Supabase ─────────────────────────────────────────────────────

def insert_batch(supabase: Client, batch: list[dict], embeddings: list[list[float]]) -> int:
    """Inserta/actualiza un lote en Supabase. Retorna número de registros insertados."""
    records = [
        {
            "chunk_id": chunk["chunk_id"],
            "content":  chunk["content"],
            "source":   chunk["source"],
            "embedding": emb,
        }
        for chunk, emb in zip(batch, embeddings)
    ]

    response = supabase.table("document_chunks").upsert(
        records, on_conflict="chunk_id"
    ).execute()

    return len(response.data)


# ── Flujo principal ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingesta PMDI → Supabase pgvector")
    parser.add_argument("--docs", type=Path, default=DOCS_DIR,
                        help="Carpeta con PDFs (default: Documentos/)")
    parser.add_argument("--from-json", action="store_true",
                        help="Usar chunks_plan_maestro.json en lugar de PDFs")
    args = parser.parse_args()

    validate_env()

    # Clientes
    nvidia_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    supabase      = create_client(SUPABASE_URL, SUPABASE_KEY)

    print("🚀 Ingesta Plan Maestro Medellín Inteligente")
    print(f"   Embeddings : {EMBED_MODEL}")
    print(f"   Supabase   : {SUPABASE_URL}")

    # Cargar chunks
    if args.from_json:
        chunks = load_chunks_from_json(CHUNKS_JSON)
    else:
        chunks = load_chunks_from_pdfs(args.docs)

    # Generar embeddings e insertar por lotes
    print(f"\n🔄 Generando embeddings e insertando (lotes de {BATCH_SIZE})…")
    inserted = 0
    failed   = 0

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), unit="lote"):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["content"] for c in batch]

        try:
            embeddings = generate_embeddings(nvidia_client, texts)
            n = insert_batch(supabase, batch, embeddings)
            inserted += n
        except Exception as e:
            print(f"\n❌ Lote {i // BATCH_SIZE + 1}: {e}")
            failed += len(batch)

    # Resumen
    print("\n" + "=" * 52)
    print("✅ INGESTA FINALIZADA")
    print(f"   Insertados : {inserted}")
    if failed:
        print(f"   Fallidos   : {failed}")
    print(f"   Total      : {len(chunks)}")
    print("=" * 52)

    # Verificación en BD
    try:
        r = supabase.table("document_chunks").select("count", count="exact").execute()
        print(f"\n✅ BD Supabase: {r.count} chunks en tabla document_chunks")
    except Exception as e:
        print(f"\n⚠️  No se pudo verificar: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Ingesta cancelada")
        sys.exit(1)
