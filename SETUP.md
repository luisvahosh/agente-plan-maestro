# 🔧 Guía de Configuración Paso a Paso

## 1️⃣ Preparación Local

### 1.1 Clonar el repositorio
```bash
git clone <repo-url> AgenteIAPlanMaestro
cd AgenteIAPlanMaestro
```

### 1.2 Crear archivo `.env`
```bash
cp .env.example .env
```

## 2️⃣ Configurar NVIDIA AI Foundation

### 2.1 Crear cuenta y obtener API Key

1. Ir a https://nvidia.com
2. Crear cuenta o login
3. Ir a NVIDIA AI Foundation
4. Crear nueva API Key
5. Copiar el formato: `nvapi-XXXX...`

### 2.2 Actualizar `.env`
```bash
NVIDIA_API_KEY=nvapi-YOUR_KEY_HERE
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
EMBED_MODEL=nvidia/nv-embedqa-e5-v5
LLM_MODEL=meta/llama-3.1-70b-instruct
```

## 3️⃣ Crear base de datos en Supabase

### 3.1 Crear proyecto Supabase

1. Ir a https://supabase.com
2. Click "New Project"
3. Seleccionar región más cercana a Hostinger
4. Esperar a que se inicie (~30-60 segundos)

### 3.2 Obtener credenciales

En **Project Settings** → **API**:
- Copiar `Project URL` → `SUPABASE_URL` en `.env`
- Copiar `anon public` key → `SUPABASE_KEY` en `.env`

Ejemplo:
```bash
SUPABASE_URL=https://abcdefgh.supabase.co
SUPABASE_KEY=eyJhbGci...
```

### 3.3 Crear tabla y extensiones

En **SQL Editor** (lado izquierdo de Supabase):
1. Click "New Query"
2. Copiar todo el contenido de `sql/init.sql`
3. Click "Run"
4. Esperar confirmación (debería decir "Query executed successfully")

La salida ejecutará:
```sql
CREATE EXTENSION vector;
CREATE TABLE document_chunks (...);
CREATE INDEX document_chunks_embedding_idx ...;
CREATE FUNCTION match_document_chunks (...);
CREATE POLICY ...;
```

### 3.4 Verificar

En **Table Editor** (izquierda) deberías ver:
- Tabla `document_chunks` con columnas: id, chunk_id, content, source, embedding, created_at

## 4️⃣ Configurar el proyecto Python

### 4.1 Instalar dependencias

```bash
# Crear venv (recomendado)
python -m venv venv

# Activar venv
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 4.2 Verificar variables de entorno

```bash
# Revisar que .env está completo
cat .env
# Debe tener NVIDIA_API_KEY, SUPABASE_URL, SUPABASE_KEY
```

## 5️⃣ Ingestar datos

### 5.1 Ejecutar ingesta

```bash
python scripts/ingest.py
```

**Salida esperada:**
```
🚀 Iniciando ingesta de datos...
📁 Chunks file: chunks_plan_maestro.json
🔑 Modelo embeddings: nvidia/nv-embedqa-e5-v5
🗄️  Supabase URL: https://...supabase.co

📂 Cargando chunks...
✓ 2435 chunks cargados

🔄 Generando embeddings (lotes de 10)...
Procesando: 100%|██████| 244/244
==================================================
✅ INGESTA COMPLETADA
  ✓ Insertados: 2435 chunks
  📊 Total: 2435 chunks
==================================================

✅ Verificación en BD: 2435 chunks en la tabla
```

**Tiempo estimado:** 10-20 minutos (depende de la velocidad de NVIDIA API)

⚠️ Si ves errores de timeout, ejecuta de nuevo. El script es idempotente (usa `ON CONFLICT` para evitar duplicados).

### 5.2 Verificar ingesta en Supabase

En Supabase **Table Editor**:
1. Click en tabla `document_chunks`
2. Deberías ver 2435 filas
3. Scroll a la derecha para ver columna `embedding` (vector[1024])

## 6️⃣ Probar localmente

### 6.1 Iniciar servidor FastAPI

```bash
uvicorn app.main:app --reload
```

**Salida esperada:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
✓ Agente IA iniciado correctamente
  - Chat web: http://localhost:8000/
  - Panel admin: http://localhost:8000/admin
  - Docs API: http://localhost:8000/docs
```

### 6.2 Probar endpoints

**Test 1: Health check**
```bash
curl http://localhost:8000/health
# Respuesta: {"status":"healthy"}
```

**Test 2: Estadísticas**
```bash
curl http://localhost:8000/admin/stats
# Respuesta: {"total_chunks":2435,"total_documents":11,"documents":[...]}
```

**Test 3: Consulta RAG**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"¿Qué es el Plan Maestro Medellín?"}'
```

**Respuesta esperada:**
```json
{
  "success": true,
  "question": "¿Qué es el Plan Maestro Medellín?",
  "answer": "El Plan Maestro Medellín... [respuesta generada con NVIDIA LLM]",
  "sources": ["PMDI Completo V4.pdf", "Anexo 4..."],
  "chunks_used": 5
}
```

### 6.3 Abrir en navegador

- Chat web: http://localhost:8000/
- Panel admin: http://localhost:8000/admin
- Documentación API: http://localhost:8000/docs

## 7️⃣ Despliegue en Hostinger

### 7.1 Preparar VPS

```bash
# SSH al servidor
ssh usuario@vps-ip

# Crear directorio
mkdir -p /home/usuario/apps/planmaestro
cd /home/usuario/apps/planmaestro

# Clonar proyecto
git clone <repo-url> .
# O con scp: scp -r . usuario@vps-ip:/home/usuario/apps/planmaestro

# Crear .env
cp .env.example .env
nano .env  # Editar con credenciales NVIDIA y Supabase
```

### 7.2 Iniciar con Docker

```bash
# Instalar Docker (si no está)
curl https://get.docker.com | sh

# Iniciar servicios
docker compose up -d

# Verificar
docker compose ps
# Debería mostrar: api, nginx con "UP"

# Ver logs
docker compose logs -f api
```

### 7.3 Verificar acceso público

```bash
# Desde otra máquina
curl http://vps-ip/health
curl http://vps-ip:80/
```

### 7.4 Configurar DNS (opcional)

Apuntar tu dominio a la IP del VPS:
- A record: `vps-ip`

Acceder a:
- http://tu-dominio/ (chat)
- http://tu-dominio/admin (panel)

## 🐛 Solución de problemas

### Error: "NVIDIA_API_KEY not configured"
**Solución:**
```bash
# Verificar .env
grep NVIDIA .env

# Debe mostrar:
# NVIDIA_API_KEY=nvapi-...

# Si está vacío, actualizar .env
nano .env
```

### Error: "Connection refused" a Supabase
**Solución:**
```bash
# Verificar SUPABASE_URL en .env
# Debe ser: https://ALGO.supabase.co
# NOT: https://localhost/...

# Verificar que la tabla existe en Supabase
# SQL Editor → SELECT COUNT(*) FROM document_chunks;
```

### Error: "Query failed on row"
**Causa:** Duplicado en ingesta (chunk_id repetido)
**Solución:** El script ya maneja esto con `ON CONFLICT`. Reintentar:
```bash
python scripts/ingest.py
```

### Ingesta muy lenta
**Soluciones:**
1. Reducir batch size en `scripts/ingest.py`:
```python
BATCH_SIZE = 5  # Cambiar de 10 a 5
```

2. Esperar pacientemente (NVIDIA API es gratuita, tiene rate limits)

### Docker no inicia
**Solución:**
```bash
# Ver error detallado
docker compose logs api

# Reiniciar
docker compose down
docker compose up -d
```

## ✅ Checklist de verificación

- [ ] Cuenta NVIDIA con API key funcionando
- [ ] Proyecto Supabase creado
- [ ] Tabla `document_chunks` creada en Supabase
- [ ] `.env` completado con credenciales
- [ ] `python scripts/ingest.py` completado con 2435 chunks
- [ ] Supabase muestra 2435 filas en tabla
- [ ] `uvicorn app.main:app --reload` inicia sin errores
- [ ] `curl http://localhost:8000/health` retorna `{"status":"healthy"}`
- [ ] Consulta RAG funciona: respuesta en español con fuentes
- [ ] Chat web carga en http://localhost:8000/
- [ ] Panel admin muestra estadísticas en http://localhost:8000/admin
- [ ] Docker compose inicia en VPS sin errores
- [ ] Endpoints públicos responden en `http://vps-ip/`

## 📞 Soporte

Si encuentras problemas:
1. Revisar los logs: `docker compose logs -f`
2. Verificar credenciales en `.env`
3. Confirmar que Supabase tiene datos: SQL Editor → `SELECT COUNT(*) FROM document_chunks;`
4. Reintentar ingesta si hay errores de API
