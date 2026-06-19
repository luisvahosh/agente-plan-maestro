# 🤖 Agente IA - Plan Maestro Medellín Inteligente

Plataforma RAG (Retrieval-Augmented Generation) para consultas inteligentes sobre el Plan Maestro Medellín Inteligente usando NVIDIA AI Foundation y Supabase pgvector.

## 🏗️ Arquitectura

```
Chat Web (Hostinger)
    ↓
FastAPI Backend (Docker)
    ↓
NVIDIA Embeddings & LLM APIs
    ↓
Supabase pgvector (BD)
```

- **Frontend**: Chat web + panel administrativo (HTML/CSS/JS puro)
- **Backend**: FastAPI + Python
- **Embeddings**: NVIDIA NIM (nvqa-e5-v5) - 1024 dimensiones
- **LLM**: NVIDIA NIM (Llama 3.1 70B) para generación
- **Base de datos**: Supabase con pgvector
- **Despliegue**: Docker en Hostinger VPS

## ⚙️ Configuración Inicial

### 1. Clonar y configurar el proyecto

```bash
git clone <repo>
cd AgenteIAPlanMaestro
cp .env.example .env
```

### 2. Completar `.env` con credenciales

```bash
# NVIDIA AI Foundation (crear cuenta en nvidia.com)
NVIDIA_API_KEY=nvapi-YOUR_KEY_HERE
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
EMBED_MODEL=nvidia/nv-embedqa-e5-v5
LLM_MODEL=meta/llama-3.1-70b-instruct

# Supabase (crear proyecto en supabase.com)
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_KEY=YOUR_ANON_KEY
```

### 3. Crear base de datos en Supabase

1. Ir a https://supabase.com
2. Crear nuevo proyecto
3. Ir al SQL Editor y ejecutar el contenido de `sql/init.sql`:
   - Crea tabla `document_chunks`
   - Instala extensión `vector`
   - Crea función `match_document_chunks` para búsqueda
   - Habilita Row Level Security

### 4. Instalar dependencias (local)

```bash
pip install -r requirements.txt
```

## 📥 Ingesta de Datos

Los 2435 fragmentos ya están extraídos en `chunks_plan_maestro.json`. Para ingresarlos a Supabase:

```bash
python scripts/ingest.py
```

Este script:
1. Carga los chunks desde JSON
2. Genera embeddings con NVIDIA (lotes de 10)
3. Inserta en Supabase con manejo de reintentos
4. Toma ~10-15 minutos para 2435 chunks

## 🚀 Ejecución Local

### Con Python puro

```bash
uvicorn app.main:app --reload
```

Acceder a:
- Chat: http://localhost:8000/
- Admin: http://localhost:8000/admin
- Docs API: http://localhost:8000/docs

### Con Docker (simula Hostinger)

```bash
docker compose up -d
```

Acceder a:
- Chat: http://localhost:80/
- Admin: http://localhost:80/admin

## 📚 Endpoints API

### POST `/query`
Consulta RAG principal
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cuáles son los ejes estratégicos del Plan Maestro?",
    "top_k": 5
  }'
```

Respuesta:
```json
{
  "success": true,
  "question": "...",
  "answer": "...",
  "sources": ["PMDI Completo V4.pdf", "..."],
  "chunks_used": 5
}
```

### GET `/admin/stats`
Estadísticas de la BD
```json
{
  "total_chunks": 2435,
  "total_documents": 11,
  "documents": ["Anexo 1...", ...]
}
```

### GET `/admin/documents`
Lista de documentos con conteo
```json
{
  "documents": [
    {"name": "PMDI Completo V4.pdf", "chunk_count": 245},
    ...
  ],
  "total": 11
}
```

### GET `/health`
Health check

### GET `/docs`
Documentación Swagger interactiva

## 🐳 Despliegue en Hostinger

### Preparación en VPS

```bash
# SSH al VPS
ssh usuario@vps-ip

# Clonar proyecto
git clone <repo> /home/usuario/apps/planmaestro
cd /home/usuario/apps/planmaestro

# Configurar .env
cp .env.example .env
# Editar .env con credenciales
nano .env

# Iniciar servicios
docker compose up -d

# Verificar
docker compose ps
docker compose logs -f api
```

### Verificar despliegue

```bash
# Test de API
curl http://vps-ip/query -X POST \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué es el PMDI?"}'

# Abrir en navegador
# Chat: http://vps-ip/
# Admin: http://vps-ip/admin
```

### Re-indexación en servidor

```bash
docker compose exec api python scripts/ingest.py
```

## 📁 Estructura

```
AgenteIAPlanMaestro/
├── app/
│   ├── main.py           # Endpoints FastAPI
│   ├── rag.py            # Pipeline RAG
│   ├── embeddings.py     # Cliente NVIDIA embeddings
│   ├── llm.py            # Cliente NVIDIA LLM
│   └── database.py       # Cliente Supabase
├── scripts/
│   ├── ingest.py         # Ingesta de chunks a Supabase
│   └── extract_pdf.py    # Extrae texto de PDFs (existente)
├── sql/
│   └── init.sql          # Inicialización de BD Supabase
├── frontend/
│   ├── index.html        # Chat web
│   └── admin.html        # Panel admin
├── nginx/
│   └── default.conf      # Configuración Nginx
├── docker-compose.yml    # Servicios Docker
├── Dockerfile            # Imagen FastAPI
├── requirements.txt      # Dependencias Python
├── chunks_plan_maestro.json  # Datos (2435 chunks)
├── Documentos/           # PDFs originales
└── README.md
```

## 🔧 Troubleshooting

### Error: "NVIDIA_API_KEY not found"
- Verificar que `.env` tiene `NVIDIA_API_KEY=nvapi-...`
- Crear cuenta en https://nvidia.com y obtener API key

### Error: "SUPABASE_URL and SUPABASE_KEY are required"
- Crear proyecto en https://supabase.com
- Obtener URL y ANON KEY desde project settings
- Ejecutar `sql/init.sql` en Supabase SQL Editor

### Error: "connection refused" en docker
- Verificar: `docker compose ps`
- Ver logs: `docker compose logs api`
- Reintentar: `docker compose restart api`

### Ingesta lenta o con timeouts
- Reducir `BATCH_SIZE` en `scripts/ingest.py` (ej: 5 en lugar de 10)
- Aumentar `MAX_RETRIES` para más intentos

## 📊 Rendimiento

- **Ingesta**: ~2435 chunks en 10-15 minutos (NVIDIA API)
- **Query**: ~2-5 segundos (embedding + búsqueda + generación)
- **BD**: pgvector indexado con IVFFLAT para búsqueda O(log N)

## 🎓 Documentación

Documentación API interactiva disponible en `/docs` (Swagger UI)

## 📝 Notas

- Los embeddings NVIDIA usan 1024 dimensiones (vs Google 768)
- Los PDFs originales están en `Documentos/` (no requeridos en runtime)
- El archivo `chunks_plan_maestro.json` es necesario para reingestión
- RLS en Supabase está permitido para demo; en producción, usar políticas más restrictivas

## 🚀 Próximas mejoras

- [ ] Autenticación de usuarios
- [ ] Historial de consultas
- [ ] Búsqueda de texto + vectorial (hybrid search)
- [ ] Upload de nuevos PDFs desde admin
- [ ] Soporte multi-idioma
- [ ] Caché de respuestas frecuentes
