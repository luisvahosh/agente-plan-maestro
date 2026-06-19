import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno ANTES de importar los modulos que crean clientes
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.rag import query
from app.database import (
    get_stats, get_chunks_by_source, delete_all_chunks,
    save_message, get_messages,
)

app = FastAPI(
    title="Agente IA - Plan Maestro Medellín",
    description="API RAG para consultar el Plan Maestro Medellín Inteligente",
    version="1.0.0"
)

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    conversation_id: str | None = None
    history: list[Message] = []

class QueryResponse(BaseModel):
    success: bool
    question: str
    answer: str
    sources: list[str] = []
    chunks_used: int = 0
    error: str = None

# Ruta a frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# Servir archivos estáticos del frontend
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Evita que los navegadores (sobre todo móviles) sirvan una versión cacheada vieja
NO_CACHE = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}

@app.get("/", tags=["Frontend"])
async def get_index():
    """Sirve la página principal (chat web)"""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file, media_type="text/html", headers=NO_CACHE)
    return {"message": "Frontend no disponible"}

@app.get("/admin", tags=["Frontend"])
async def get_admin():
    """Sirve el panel administrativo"""
    admin_file = FRONTEND_DIR / "admin.html"
    if admin_file.exists():
        return FileResponse(admin_file, media_type="text/html", headers=NO_CACHE)
    return {"message": "Panel admin no disponible"}

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def post_query(request: QueryRequest) -> QueryResponse:
    """
    Endpoint principal: realiza una consulta RAG sobre el Plan Maestro.

    Args:
        question: La pregunta del usuario
        top_k: Número de chunks a recuperar (default: 5)

    Returns:
        Respuesta con el texto generado y las fuentes consultadas
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")

    history = [{"role": m.role, "content": m.content} for m in request.history]
    result = await query(request.question, history=history, top_k=request.top_k)

    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("error", "Error desconocido"))

    # Guardar la conversación en Supabase (si el cliente envió un id)
    if request.conversation_id:
        await save_message(request.conversation_id, "user", request.question)
        await save_message(
            request.conversation_id, "assistant",
            result["answer"], result.get("sources", []),
        )

    return QueryResponse(
        success=result["success"],
        question=result["question"],
        answer=result["answer"],
        sources=result.get("sources", []),
        chunks_used=result.get("chunks_used", 0)
    )

@app.get("/conversation/{conversation_id}", tags=["RAG"])
async def get_conversation(conversation_id: str) -> dict:
    """Recupera los mensajes guardados de una conversación."""
    messages = await get_messages(conversation_id)
    return {"conversation_id": conversation_id, "messages": messages}

@app.get("/admin/stats", tags=["Admin"])
async def get_admin_stats() -> dict:
    """
    Obtiene estadísticas de la BD:
    - Total de chunks indexados
    - Total de documentos
    - Lista de documentos
    """
    stats = await get_stats()
    return stats

@app.get("/admin/documents", tags=["Admin"])
async def get_admin_documents() -> dict:
    """
    Obtiene lista de documentos con conteo de chunks.

    Returns:
        Lista de documentos con número de chunks cada uno
    """
    chunks_by_source = await get_chunks_by_source()
    documents = [
        {
            "name": source,
            "chunk_count": count
        }
        for source, count in sorted(chunks_by_source.items())
    ]
    return {"documents": documents, "total": len(documents)}

@app.post("/admin/reindex", tags=["Admin"])
async def post_reindex() -> dict:
    """
    Limpia la BD actual y ejecuta reingestión desde chunks_plan_maestro.json.

    NOTA: Esta operación requiere ejecutar scripts/ingest.py manualmente en el servidor.
    Este endpoint solo informa el estado.
    """
    return {
        "status": "pending",
        "message": "Para reingestionar, ejecuta: python scripts/ingest.py en el servidor",
        "steps": [
            "1. Ejecutar: python scripts/ingest.py",
            "2. Esperar a que se completen los embeddings y la inserción",
            "3. Verificar con GET /admin/stats"
        ]
    }

@app.get("/health", tags=["System"])
async def health_check() -> dict:
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/info", tags=["System"])
async def get_info() -> dict:
    """Información de la aplicación"""
    return {
        "name": "Agente IA - Plan Maestro Medellín",
        "version": "1.0.0",
        "description": "API RAG para consultar el Plan Maestro Medellín Inteligente"
    }

@app.on_event("startup")
async def startup():
    print("[OK] Agente IA iniciado correctamente")
    print("  - Chat web    : http://localhost:8000/")
    print("  - Panel admin : http://localhost:8000/admin")
    print("  - Docs API    : http://localhost:8000/docs")
