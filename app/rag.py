import asyncio

from app.embeddings import embed_query
from app.llm import generate_answer
from app.database import search_similar, keyword_search, search_verified

# Umbral minimo de similitud vectorial para considerar una pregunta "dentro del tema".
# Si la mejor coincidencia vectorial no lo supera, la pregunta se considera fuera
# del contexto de los PDFs y NO se llama al LLM (evita alucinacion).
MIN_SIMILARITY = 0.42

# Cuantos chunks recuperar de cada fuente
VECTOR_TOP_K = 10          # red amplia para que el chunk relevante aparezca
KEYWORD_LIMIT = 4
# Menos chunks al contexto = menos tokens de entrada = el 70b responde más rápido.
# El banco verificado se inyecta aparte (search_verified), así que no se pierde precisión.
MAX_CONTEXT_CHUNKS = 6

# Fuentes curadas/verificadas por el equipo PMDI. Tienen PRIORIDAD en el contexto:
# son respuestas revisadas, más precisas que los fragmentos crudos de los PDFs.
VERIFIED_SOURCES = {"Banco Q&A verificado PMDI", "Casos de articulación PMDI"}

NO_CONTEXT_RESPONSE = (
    "No encontré información sobre esto en los documentos del "
    "Plan Maestro Medellín Inteligente. Por favor, formula una "
    "pregunta relacionada con el contenido del plan."
)


def build_context(chunks: list[dict]) -> str:
    """Construye el contexto numerado a partir de los chunks recuperados."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Desconocido")
        content = chunk.get("content", "").strip()
        parts.append(f"[Fragmento {i} — {source}]\n{content}")
    return "\n\n".join(parts)


def extract_sources(chunks: list[dict]) -> list[str]:
    """Retorna lista de fuentes únicas ordenada."""
    return sorted({c.get("source", "") for c in chunks if c.get("source")})


def _best_similarity(chunks: list[dict]) -> float:
    if not chunks:
        return 0.0
    return max(c.get("similarity", 0.0) for c in chunks)


def _merge_chunks(vector_chunks: list[dict], keyword_chunks: list[dict]) -> list[dict]:
    """
    Combina los resultados priorizando:
    1. Chunks de FUENTES VERIFICADAS (banco Q&A curado) — van primero, porque son
       respuestas revisadas y precisas. Esto evita que al reformular la pregunta el
       agente caiga en fragmentos crudos de PDF y pierda los datos exactos.
    2. Mejores chunks por palabra clave (término exacto).
    3. Chunks vectoriales (semántica).
    """
    seen = set()
    result = []

    def add(chunk):
        cid = chunk.get("chunk_id")
        if cid not in seen and len(result) < MAX_CONTEXT_CHUNKS:
            seen.add(cid)
            result.append(chunk)

    # 1. PRIORIDAD: chunks verificados que aparezcan en cualquiera de las búsquedas
    for c in vector_chunks + keyword_chunks:
        if c.get("source") in VERIFIED_SOURCES:
            add(c)

    # 2. Mejores chunks por palabra clave
    keyword_slots = MAX_CONTEXT_CHUNKS // 2
    for c in keyword_chunks[:keyword_slots]:
        add(c)

    # 3. Rellenar con vectoriales (semántica)
    for c in vector_chunks:
        add(c)

    # 4. Resto de palabra clave si queda espacio
    for c in keyword_chunks[keyword_slots:]:
        add(c)

    return result


async def prepare(question: str, history: list[dict] | None = None,
                  top_k: int = VECTOR_TOP_K) -> dict:
    """
    Hace SOLO la recuperación (embedding + búsquedas + merge) y decide si la pregunta
    está fuera de tema. Devuelve el contexto y las fuentes listos para generar.
    Se usa tanto en la respuesta normal como en la de streaming.
    """
    question_embedding = embed_query(question)

    # Las tres búsquedas son independientes → se ejecutan EN PARALELO (más rápido)
    vector_chunks, keyword_chunks, verified_chunks = await asyncio.gather(
        search_similar(question_embedding, top_k=top_k, threshold=0.0),
        keyword_search(question, limit=KEYWORD_LIMIT),
        search_verified(question_embedding, top_k=2, threshold=0.45),
    )

    # Barrera anti-alucinación: si la pregunta está fuera de tema y no hay historial.
    # Si hay un chunk verificado relevante, también se considera dentro de tema.
    best_sim = _best_similarity(vector_chunks)
    if best_sim < MIN_SIMILARITY and not verified_chunks and not history:
        return {"off_topic": True, "answer": NO_CONTEXT_RESPONSE,
                "sources": [], "context": "", "best_similarity": best_sim}

    chunks = _merge_chunks(verified_chunks + vector_chunks, keyword_chunks)

    return {"off_topic": False, "answer": None, "sources": extract_sources(chunks),
            "context": build_context(chunks), "best_similarity": best_sim,
            "chunks_used": len(chunks)}


async def query(question: str, history: list[dict] | None = None,
                top_k: int = VECTOR_TOP_K) -> dict:
    """
    Pipeline RAG híbrido con restricción estricta al contexto de los PDFs
    y memoria conversacional.

    Flujo:
    1. Embedding de la pregunta + búsqueda vectorial (semántica)
    2. Barrera anti-alucinación: si la mejor similitud < MIN_SIMILARITY y NO hay
       historial, la pregunta está fuera de tema → respuesta "sin información".
       (En preguntas de seguimiento se confía en el historial + el prompt estricto.)
    3. Búsqueda por palabra clave (complementa la semántica para términos exactos)
    4. Se combinan ambos resultados y se genera la respuesta con el LLM (con historial)
    """
    try:
        prep = await prepare(question, history=history, top_k=top_k)

        if prep["off_topic"]:
            return {
                "success": True, "question": question,
                "answer": prep["answer"], "sources": [], "chunks_used": 0,
                "best_similarity": prep["best_similarity"],
            }

        answer = generate_answer(question, prep["context"], history)
        return {
            "success": True, "question": question,
            "answer": answer, "sources": prep["sources"],
            "chunks_used": prep.get("chunks_used", 0),
            "best_similarity": prep["best_similarity"],
        }

    except Exception as e:
        return {
            "success": False,
            "question": question,
            "error": str(e),
            "answer": "Ocurrió un error al procesar tu pregunta. Por favor, intenta de nuevo.",
        }
