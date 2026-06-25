import os
import re
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY are required")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Palabras vacias en espanol que NO aportan a la busqueda por palabra clave
STOPWORDS = {
    "cuantos", "cuantas", "cuales", "cual", "como", "donde", "quien", "quienes",
    "tiene", "tienen", "para", "por", "del", "los", "las", "con", "una", "uno",
    "que", "este", "esta", "estos", "estas", "son", "sobre", "entre", "desde",
    "plan", "maestro", "medellin", "distrito", "inteligente", "pmdi", "hay",
    "cuanto", "cuanta", "segun", "dentro", "entonces", "tambien", "porque",
}


async def keyword_search(question: str, limit: int = 5) -> list[dict]:
    """
    Busca chunks que contengan las palabras clave de la pregunta y los ORDENA
    por relevancia (cuántas veces aparecen los términos en el chunk).
    Complementa la busqueda vectorial cuando un termino exacto importa (ej: 'pilares').
    """
    words = [
        w for w in re.findall(r"[a-záéíóúñ]+", question.lower())
        if len(w) >= 4 and w not in STOPWORDS
    ]
    if not words:
        return []

    or_filter = ",".join(f"content.ilike.%{w}%" for w in words)
    try:
        # Traer más candidatos (50) para luego rankear por relevancia en Python
        result = (
            supabase.table("document_chunks")
            .select("id,chunk_id,content,source")
            .or_(or_filter)
            .limit(50)
            .execute()
        )
        candidates = result.data or []

        # Puntuar cada chunk: suma de apariciones de cada palabra clave
        def score(chunk):
            text = chunk["content"].lower()
            return sum(text.count(w) for w in words)

        candidates.sort(key=score, reverse=True)
        top = candidates[:limit]
        for c in top:
            c["similarity"] = 0.5  # marcador neutro de relevancia por palabra clave
        return top
    except Exception as e:
        print(f"Error in keyword search: {e}")
        return []

async def search_similar(embedding: list[float], top_k: int = 5, threshold: float = 0.0) -> list[dict]:
    """
    Busca chunks similares usando pgvector en Supabase.
    Retorna los chunks con mayor similitud.
    """
    try:
        response = supabase.rpc(
            'match_document_chunks',
            {
                'query_embedding': embedding,
                'match_threshold': threshold,
                'match_count': top_k
            }
        ).execute()

        return response.data if response.data else []
    except Exception as e:
        print(f"Error searching similar chunks: {e}")
        return []

def _fetch_all_sources() -> list[str]:
    """
    Trae la columna 'source' de TODOS los chunks paginando.
    Supabase limita cada select a 1000 filas, por eso se usa range().
    """
    sources: list[str] = []
    page = 0
    page_size = 1000
    while True:
        start = page * page_size
        end = start + page_size - 1
        result = supabase.table('document_chunks').select('source').range(start, end).execute()
        if not result.data:
            break
        sources.extend(item['source'] for item in result.data)
        if len(result.data) < page_size:
            break
        page += 1
    return sources


# ── Búsqueda dedicada al banco verificado (solo 22 chunks, se cachea) ──────────
import json as _json

_verified_cache: list[dict] | None = None
VERIFIED_MIN_CHUNK_ID = 1_000_000  # rango reservado para Q&A/casos verificados


def _load_verified_chunks() -> list[dict]:
    """Carga (una vez) los chunks verificados con su embedding parseado a lista."""
    global _verified_cache
    if _verified_cache is not None:
        return _verified_cache
    try:
        r = (supabase.table("document_chunks")
             .select("chunk_id,content,source,embedding")
             .gte("chunk_id", VERIFIED_MIN_CHUNK_ID)
             .execute())
        out = []
        for row in (r.data or []):
            emb = row.get("embedding")
            if isinstance(emb, str):
                emb = _json.loads(emb)
            row["embedding"] = emb
            out.append(row)
        _verified_cache = out
    except Exception as e:
        print(f"Error loading verified chunks: {e}")
        _verified_cache = []
    return _verified_cache


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


async def search_verified(query_embedding: list[float], top_k: int = 2,
                          threshold: float = 0.5) -> list[dict]:
    """
    Busca SOLO entre los chunks verificados (banco Q&A curado) por similitud coseno.
    Garantiza que la mejor respuesta curada entre al contexto aunque, al reformular
    la pregunta, no destaque frente a los 2420 chunks de los PDFs.
    """
    chunks = _load_verified_chunks()
    scored = []
    for c in chunks:
        sim = _cosine(query_embedding, c["embedding"])
        if sim >= threshold:
            scored.append({"chunk_id": c["chunk_id"], "content": c["content"],
                           "source": c["source"], "similarity": sim})
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]


async def get_stats() -> dict:
    """Obtiene estadísticas de la base de datos."""
    try:
        # Total de chunks (conteo exacto, no limitado por las 1000 filas)
        result = supabase.table('document_chunks').select('id', count='exact').limit(1).execute()
        total_chunks = result.count or 0

        # Documentos únicos (paginando para no perder ninguno)
        unique_sources = sorted(set(_fetch_all_sources()))

        return {
            'total_chunks': total_chunks,
            'total_documents': len(unique_sources),
            'documents': unique_sources,
        }
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {'total_chunks': 0, 'total_documents': 0, 'documents': []}


async def get_chunks_by_source() -> dict:
    """Obtiene conteo de chunks por documento (paginado)."""
    try:
        chunks_by_source: dict[str, int] = {}
        for source in _fetch_all_sources():
            chunks_by_source[source] = chunks_by_source.get(source, 0) + 1
        return chunks_by_source
    except Exception as e:
        print(f"Error getting chunks by source: {e}")
        return {}

async def delete_all_chunks():
    """Elimina todos los chunks de la BD (para reingestión)"""
    try:
        supabase.table('document_chunks').delete().neq('id', None).execute()
        return {"status": "success", "message": "All chunks deleted"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ──────────────────────────────────────────────────────────────────────────
# Historial de conversaciones (tabla chat_messages)
# ──────────────────────────────────────────────────────────────────────────

async def save_message(conversation_id: str, role: str, content: str,
                       sources: list[str] | None = None) -> None:
    """Guarda un mensaje del chat en Supabase."""
    try:
        supabase.table("chat_messages").insert({
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "sources": sources or [],
        }).execute()
    except Exception as e:
        print(f"Error saving message: {e}")


async def get_messages(conversation_id: str, limit: int = 200) -> list[dict]:
    """Recupera los mensajes de una conversación en orden cronológico."""
    try:
        result = (
            supabase.table("chat_messages")
            .select("role,content,sources,created_at")
            .eq("conversation_id", conversation_id)
            .order("created_at")
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"Error getting messages: {e}")
        return []
