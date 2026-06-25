#!/usr/bin/env python3
"""
Aumento de "entrenamiento" del agente (data augmentation para RAG).

Toma las variantes parafraseadas (entrenamiento/preguntas_variantes.json) y las
inyecta al banco verificado de Supabase como pares pregunta+respuesta. Así el agente
reconoce MUCHAS más formas de preguntar lo mismo y recupera la respuesta curada
aunque el usuario lo formule distinto (mejora la generalización).

Cada variante -> chunk verificado (source "Banco Q&A verificado PMDI") con la
respuesta de referencia original. chunk_id en rango reservado 1_200_000+.

Requiere: python scripts/generar_variantes.py (genera el JSON de variantes)
Uso:      python scripts/aumentar_entrenamiento.py
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()

from openai import OpenAI
from supabase import create_client, Client

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

ENTR = ROOT / "entrenamiento"
VARIANTES_JSON = ENTR / "preguntas_variantes.json"

SOURCE = "Banco Q&A verificado PMDI"   # se trata como conocimiento verificado
CHUNK_ID_BASE = 1_200_000              # rango reservado para variantes de entrenamiento
BATCH = 10
MAX_RETRIES = 4

client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def embeddings(texts):
    for a in range(MAX_RETRIES):
        try:
            r = client.embeddings.create(input=texts, model=EMBED_MODEL,
                                         extra_body={"input_type": "passage", "truncate": "END"})
            return [i.embedding for i in r.data]
        except Exception as e:
            if a < MAX_RETRIES - 1:
                time.sleep(2 ** (a + 1))
            else:
                raise


def main():
    if not VARIANTES_JSON.exists():
        print(f"❌ Falta {VARIANTES_JSON}. Corre primero: python scripts/generar_variantes.py")
        sys.exit(1)

    data = json.load(open(VARIANTES_JSON, encoding="utf-8"))
    chunks = []
    for item in data:
        ref = item["referencia"].strip()
        for v in item["variantes"]:
            v = (v or "").strip()
            if not v:
                continue
            content = f"[{SOURCE}]\nPregunta: {v}\nRespuesta: {ref}"
            chunks.append({
                "chunk_id": CHUNK_ID_BASE + len(chunks),
                "content": content.replace("\x00", ""),
                "source": SOURCE,
            })

    print(f"🧠 Inyectando {len(chunks)} variantes como conocimiento verificado…")
    inserted = 0
    for i in range(0, len(chunks), BATCH):
        b = chunks[i:i + BATCH]
        embs = embeddings([c["content"] for c in b])
        records = [{"chunk_id": c["chunk_id"], "content": c["content"],
                    "source": c["source"], "embedding": e} for c, e in zip(b, embs)]
        r = supabase.table("document_chunks").upsert(records, on_conflict="chunk_id").execute()
        inserted += len(r.data)
        print(f"  lote {i // BATCH + 1}: +{len(r.data)}")

    # total verificados ahora
    tot = supabase.table("document_chunks").select("chunk_id", count="exact").gte("chunk_id", 1_000_000).execute()
    print(f"\n✅ {inserted} variantes inyectadas. Banco verificado total: {tot.count} chunks.")
    print("   (Reinicia el contenedor/servidor para refrescar la caché de search_verified.)")


if __name__ == "__main__":
    main()
