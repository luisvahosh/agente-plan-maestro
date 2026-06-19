from app.embeddings import embed_query
from app.llm import generate_answer
from app.database import search_similar, keyword_search

# Umbral minimo de similitud vectorial para considerar una pregunta "dentro del tema".
# Si la mejor coincidencia vectorial no lo supera, la pregunta se considera fuera
# del contexto de los PDFs y NO se llama al LLM (evita alucinacion).
MIN_SIMILARITY = 0.42

# Cuantos chunks recuperar de cada fuente
VECTOR_TOP_K = 6
KEYWORD_LIMIT = 5
MAX_CONTEXT_CHUNKS = 7

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
    Combina garantizando representación de AMBAS búsquedas:
    reserva ~la mitad de los slots para los mejores chunks por palabra clave
    (ya rankeados) y la otra mitad para los vectoriales. Así el chunk con el
    término exacto (ej: 'cinco pilares') no queda desplazado por los semánticos.
    """
    keyword_slots = MAX_CONTEXT_CHUNKS // 2  # p.ej. 3 de 7
    seen = set()
    result = []

    # 1. Mejores chunks por palabra clave (relevancia exacta)
    for c in keyword_chunks[:keyword_slots]:
        cid = c.get("chunk_id")
        if cid not in seen:
            seen.add(cid)
            result.append(c)

    # 2. Rellenar con chunks vectoriales (semántica)
    for c in vector_chunks:
        if len(result) >= MAX_CONTEXT_CHUNKS:
            break
        cid = c.get("chunk_id")
        if cid not in seen:
            seen.add(cid)
            result.append(c)

    # 3. Si queda espacio, más chunks por palabra clave
    for c in keyword_chunks[keyword_slots:]:
        if len(result) >= MAX_CONTEXT_CHUNKS:
            break
        cid = c.get("chunk_id")
        if cid not in seen:
            seen.add(cid)
            result.append(c)

    return result


async def query(question: str, top_k: int = VECTOR_TOP_K) -> dict:
    """
    Pipeline RAG híbrido con restricción estricta al contexto de los PDFs.

    Flujo:
    1. Embedding de la pregunta + búsqueda vectorial (semántica)
    2. Barrera anti-alucinación: si la mejor similitud < MIN_SIMILARITY,
       la pregunta está fuera de tema → respuesta "sin información" (no llama al LLM)
    3. Búsqueda por palabra clave (complementa la semántica para términos exactos)
    4. Se combinan ambos resultados y se genera la respuesta con el LLM
    """
    try:
        # 1. Búsqueda vectorial
        question_embedding = embed_query(question)
        vector_chunks = await search_similar(question_embedding, top_k=top_k, threshold=0.0)

        # 2. Barrera anti-alucinación (rechaza preguntas claramente fuera de tema)
        best_sim = _best_similarity(vector_chunks)
        if best_sim < MIN_SIMILARITY:
            return {
                "success": True,
                "question": question,
                "answer": NO_CONTEXT_RESPONSE,
                "sources": [],
                "chunks_used": 0,
                "best_similarity": best_sim,
            }

        # 3. Búsqueda por palabra clave (híbrida)
        keyword_chunks = await keyword_search(question, limit=KEYWORD_LIMIT)

        # 4. Combinar y generar respuesta
        chunks = _merge_chunks(vector_chunks, keyword_chunks)
        context = build_context(chunks)
        answer = generate_answer(question, context)
        sources = extract_sources(chunks)

        return {
            "success": True,
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_used": len(chunks),
            "best_similarity": best_sim,
        }

    except Exception as e:
        return {
            "success": False,
            "question": question,
            "error": str(e),
            "answer": "Ocurrió un error al procesar tu pregunta. Por favor, intenta de nuevo.",
        }
