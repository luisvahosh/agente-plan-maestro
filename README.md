# 🤖 Asistente Virtual del PMDI

Agente conversacional (RAG) que responde preguntas sobre el **Plan Maestro Medellín Distrito Inteligente (PMDI)**, basándose únicamente en los documentos oficiales del plan. Identidad visual de la Alcaldía de Medellín.

🌐 En producción: **https://agentepmdi.centrodepensamientoitm.cloud/**

---

## Arquitectura

```
Usuario (web/móvil)
   │
   ▼
Nginx Proxy Manager (HTTPS, dominio)
   │
   ▼
FastAPI (Docker en VPS Hostinger)
   │   1. Embedding de la pregunta ─────► NVIDIA Embeddings API
   │   2. Búsqueda híbrida (paralela):
   │        • vectorial (pgvector)
   │        • palabra clave
   │        • banco verificado (curado)
   │   3. Genera respuesta (streaming) ──► NVIDIA LLM API
   ▼
Supabase (PostgreSQL + pgvector)
   • document_chunks  (PDFs + banco Q&A verificado)
   • chat_messages    (historial de conversaciones)
```

**Tecnologías**
- **Embeddings**: NVIDIA `nvidia/nv-embedqa-e5-v5` (1024 dimensiones)
- **LLM**: NVIDIA `meta/llama-3.1-8b-instruct` (rápido; configurable en `.env`)
- **Base de datos**: Supabase (pgvector) — ~2420 chunks de PDFs + banco Q&A verificado
- **Backend**: FastAPI (Python) con respuestas en streaming (SSE)
- **Frontend**: HTML/CSS/JS, identidad Alcaldía de Medellín, móvil-first
- **Despliegue**: Docker en VPS Hostinger, detrás de Nginx Proxy Manager

---

## Estructura del proyecto

```
app/
  main.py        Endpoints FastAPI (/query, /query-stream, /admin/*, /conversation/*)
  rag.py         Pipeline RAG (recuperación híbrida + barrera anti-alucinación)
  embeddings.py  Cliente NVIDIA embeddings (input_type query/passage)
  llm.py         Cliente NVIDIA LLM (system prompt institucional + streaming)
  database.py    Supabase: búsqueda vectorial, banco verificado, historial
frontend/
  index.html     Chat web (streaming, membrete de fuentes con chips)
  admin.html     Panel administrativo (estadísticas, documentos)
  logo-alcaldia.png
scripts/
  ingest.py                 Ingesta de los PDFs de Documentos/ a Supabase
  ingest_qa.py              Inyecta el banco Q&A verificado (entrenamiento/)
  aumentar_entrenamiento.py Inyecta variantes parafraseadas (data augmentation)
  generar_variantes.py      Genera variantes para evaluar generalización
  evaluar_agente.py         Batería de evaluación (golden set + LLM-juez)
  evaluar_variantes.py      Evaluación de generalización (variantes)
sql/
  init.sql            Tabla document_chunks + función match_document_chunks
  chat_messages.sql   Tabla del historial de chat
  create_index.sql    Índice IVFFLAT (correr tras la ingesta)
Documentos/      12 PDFs del PMDI (fuente, local)
entrenamiento/   Banco Q&A verificado y reportes de evaluación (local)
Dockerfile · docker-compose.yml · nginx/ · requirements.txt · .env
```

---

## Variables de entorno (`.env`)

```bash
NVIDIA_API_KEY=nvapi-...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
EMBED_MODEL=nvidia/nv-embedqa-e5-v5
LLM_MODEL=meta/llama-3.1-8b-instruct      # 70b es más preciso pero MUY lento en el tier gratuito
SUPABASE_URL=https://TU-PROYECTO.supabase.co
SUPABASE_KEY=eyJ...                         # service_role key
```

---

## Puesta en marcha rápida

Ver **SETUP.md** para el paso a paso completo. Resumen:

```bash
# 1. Dependencias (local)
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# 2. Crear tablas en Supabase (SQL Editor): sql/init.sql y sql/chat_messages.sql

# 3. Ingesta de los PDFs + banco verificado (una vez, desde local)
python scripts/ingest.py
python scripts/ingest_qa.py

# 4. Probar local
uvicorn app.main:app --reload     # http://localhost:8000/
```

## Despliegue / actualización en el VPS

```bash
cd ~/agente-plan-maestro
git pull
docker compose up -d --build
```
El `LLM_MODEL` se cambia en el `.env` del VPS (no va en git). El acceso público va por Nginx Proxy Manager → contenedor `agente-plan-maestro:8000`.

---

## Notas

- **Restricción de contexto**: el agente solo responde con base en los documentos del PMDI. Si la pregunta es ajena, redirige cortésmente (barrera por umbral de similitud + prompt estricto).
- **Banco verificado**: respuestas curadas por el equipo del PMDI (con su nombre propio en las citas), priorizadas sobre los fragmentos crudos de los PDFs.
- **Velocidad**: la latencia depende del tier gratuito de NVIDIA (variable). El streaming muestra la respuesta progresivamente para mejorar la experiencia.
- **Evaluación**: `scripts/evaluar_agente.py` (golden set) y `scripts/evaluar_variantes.py` (generalización con paráfrasis).
