# 📘 Documentación completa — Asistente Virtual del PMDI

Guía paso a paso de **todo lo que se construyó**: lo funcional, lo técnico, lo que se hizo en **Supabase**, con el **modelo de NVIDIA** y lo que se desplegó en **Hostinger**. Al final hay una sección de **portabilidad** para reusar este conocimiento en **Azure, Gemini o DeepSeek**.

---

## ÍNDICE
1. ¿Qué es y qué resuelve? (funcional)
2. Cómo funciona un agente RAG (concepto base)
3. Arquitectura general
4. Paso A — Base de datos vectorial (Supabase + pgvector)
5. Paso B — Modelo de IA (NVIDIA: embeddings y LLM)
6. Paso C — Ingesta de documentos (PDFs → vectores)
7. Paso D — Banco verificado y "entrenamiento"
8. Paso E — El pipeline RAG (cómo responde)
9. Paso F — Backend (FastAPI) y streaming
10. Paso G — Frontend (chat, memoria, identidad)
11. Paso H — Despliegue en Hostinger (Docker + Nginx Proxy Manager)
12. Portabilidad: cambiar a Azure / Gemini / DeepSeek
13. Decisiones y aprendizajes clave (los "tropiezos")
14. Glosario

---

## 1. ¿Qué es y qué resuelve? (funcional)

Es un **asistente conversacional** que responde preguntas en lenguaje natural sobre el **Plan Maestro Medellín Distrito Inteligente (PMDI)**, usando **únicamente** los documentos oficiales del plan (12 PDFs + un banco de respuestas verificadas).

**Lo importante a nivel funcional:**
- No inventa: si la información no está en los documentos, lo dice o redirige.
- Cita las fuentes de cada respuesta.
- Recuerda la conversación (se le puede preguntar "y el segundo pilar?").
- Distingue contenido **verificado** (curado por el equipo) de los PDFs crudos.
- Funciona en web y móvil, con la identidad visual de la Alcaldía de Medellín.

---

## 2. Cómo funciona un agente RAG (concepto base)

**RAG = Retrieval-Augmented Generation** (generación aumentada por recuperación). Es el patrón que hace que un modelo de lenguaje responda sobre TUS documentos sin "entrenarlo".

La idea, en 3 pasos:
1. **Indexar** (una vez): se parten los documentos en fragmentos ("chunks"), se convierte cada fragmento en un **vector** (lista de números que representa su significado) y se guardan en una **base de datos vectorial**.
2. **Recuperar** (en cada pregunta): se convierte la pregunta en un vector y se buscan los fragmentos **más parecidos** (búsqueda por similitud).
3. **Generar**: se le entrega al **modelo de lenguaje (LLM)** la pregunta + los fragmentos encontrados, y se le pide responder **solo con eso**.

> Clave: el LLM no "sabe" del PMDI; se le **da el contexto** en cada pregunta. Por eso se puede cambiar de modelo sin re-entrenar nada.

Dos modelos distintos intervienen:
- **Modelo de embeddings**: convierte texto → vector (para indexar y para buscar).
- **Modelo de lenguaje (LLM)**: redacta la respuesta a partir del contexto.

---

## 3. Arquitectura general

```
Usuario (navegador / móvil)
      │  HTTPS
      ▼
Nginx Proxy Manager  ── dominio + certificado SSL
      │
      ▼
FastAPI (contenedor Docker en el VPS Hostinger)
      │
      │  1) Pregunta → vector ───────────► API de embeddings (NVIDIA)
      │  2) Buscar fragmentos parecidos ─► Supabase (pgvector)
      │  3) Pregunta + contexto → texto ─► API del LLM (NVIDIA)
      ▼
Respuesta + fuentes (en streaming)

Supabase (PostgreSQL + pgvector):
  • document_chunks → fragmentos de PDFs + banco verificado (con su vector)
  • chat_messages   → historial de conversaciones
```

**Piezas y su rol (esto es lo portable):**
| Pieza | En este proyecto | Se puede cambiar por |
|---|---|---|
| Modelo de embeddings | NVIDIA `nv-embedqa-e5-v5` (1024 dim) | Azure OpenAI, Gemini, etc. |
| Base de datos vectorial | Supabase (PostgreSQL + pgvector) | Azure AI Search, Postgres, Pinecone, Qdrant |
| Modelo de lenguaje (LLM) | NVIDIA `llama-3.1-8b-instruct` | Azure OpenAI (GPT), Gemini, DeepSeek |
| Backend | FastAPI (Python) | igual |
| Hosting | Docker en VPS Hostinger | Azure Container Apps, etc. |
| Proxy/HTTPS | Nginx Proxy Manager | Azure Front Door, Nginx, Caddy |

---

## 4. Paso A — Base de datos vectorial (Supabase + pgvector)

**Qué es:** Supabase es un PostgreSQL gestionado en la nube. La extensión **pgvector** le agrega un tipo de dato `vector` y búsqueda por similitud. Es donde viven los fragmentos y sus vectores.

**Pasos que se hicieron:**
1. Crear un proyecto en supabase.com.
2. En el **SQL Editor**, ejecutar `sql/init.sql`:
   - `CREATE EXTENSION vector;` (activa pgvector)
   - Tabla `document_chunks`:
     ```sql
     CREATE TABLE document_chunks (
       id BIGSERIAL PRIMARY KEY,
       chunk_id INTEGER UNIQUE,     -- id estable del fragmento
       content TEXT,                -- el texto del fragmento
       source TEXT,                 -- de qué documento viene (para citar)
       embedding vector(1024),      -- el vector (1024 = dimensión del modelo NVIDIA)
       created_at TIMESTAMPTZ DEFAULT NOW()
     );
     ```
   - Función `match_document_chunks(...)`: hace la búsqueda por **similitud coseno** (`1 - (embedding <=> query)`) y devuelve los N fragmentos más parecidos.
   - **RLS** (Row Level Security): políticas para permitir lectura pública e inserción con la `service_role` key.
3. Ejecutar `sql/chat_messages.sql`: tabla del historial del chat (`conversation_id`, `role`, `content`, `sources`, `created_at`).
4. (Opcional, tras la ingesta) `sql/create_index.sql`: índice `ivfflat` para acelerar la búsqueda cuando hay muchos vectores.

> ⚠️ **La dimensión del vector (1024) DEBE coincidir con el modelo de embeddings.** Si cambias de modelo, cambia ese número y re-ingestas (ver Portabilidad).

---

## 5. Paso B — Modelo de IA (NVIDIA: embeddings y LLM)

**Qué se usó:** la plataforma gratuita **build.nvidia.com**, que expone modelos con una API **compatible con OpenAI** (mismo formato de llamada que la API de OpenAI).

- **Embeddings:** `nvidia/nv-embedqa-e5-v5` → vector de **1024 dimensiones**.
  - Detalle crítico: NVIDIA exige un parámetro `input_type`: `"passage"` para indexar documentos y `"query"` para la pregunta del usuario. Sin él, la API falla.
- **LLM:** `meta/llama-3.1-8b-instruct` (rápido). Se probó `llama-3.3-70b-instruct` (más preciso pero muy lento en el tier gratuito).

**Cómo se llama (cliente OpenAI reutilizado):**
```python
from openai import OpenAI
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1",
                api_key="nvapi-...")

# Embedding
client.embeddings.create(input=[texto], model="nvidia/nv-embedqa-e5-v5",
                         extra_body={"input_type": "passage", "truncate": "END"})

# Chat
client.chat.completions.create(model="meta/llama-3.1-8b-instruct",
                               messages=[...], temperature=0.1, stream=True)
```
> Como se usó el **cliente de OpenAI** apuntando a NVIDIA, cambiar de proveedor suele ser solo cambiar `base_url`, `api_key` y el nombre del modelo (ver Portabilidad).

---

## 6. Paso C — Ingesta de documentos (PDFs → vectores)

Script: `scripts/ingest.py`. Se corre **una vez** desde tu máquina.

Flujo:
1. Lee cada PDF de `Documentos/` con **PyMuPDF**.
2. Divide el texto en **chunks de 1000 caracteres con 200 de solapamiento** (el solapamiento evita cortar ideas a la mitad).
3. Limpia caracteres nulos `\x00` (PostgreSQL los rechaza).
4. Genera el embedding de cada chunk con NVIDIA (`input_type="passage"`), en **lotes de 10** con reintentos.
5. Inserta en Supabase con **upsert por `chunk_id`** (idempotente: re-correrlo no duplica).

Resultado: ~**2420 fragmentos** vectorizados en Supabase.

---

## 7. Paso D — Banco verificado y "entrenamiento"

Un RAG con API **no se puede re-entrenar (fine-tuning)**, pero se "entrena" mejorando el **conocimiento curado** y la **recuperación**:

1. **Banco Q&A verificado** (`scripts/ingest_qa.py`): se inyectaron 12 preguntas-respuesta verificadas + 10 casos, cada chunk con la **pregunta delante** (técnica *semantic Q&A matching*). Así, si alguien pregunta algo parecido, se recupera la **respuesta curada**. Se guardan con `chunk_id` en un rango reservado (1.000.000+) y un `source` propio ("Banco Q&A verificado PMDI") para distinguirlas y **priorizarlas**.
2. **Aumento con variantes** (`scripts/aumentar_entrenamiento.py`): se generaron paráfrasis de cada pregunta (`generar_variantes.py`) y se inyectaron como banco verificado (66 chunks en total). Esto hace que **más formas de preguntar** lleguen a la respuesta correcta.
3. **Evaluación** (`evaluar_agente.py`, `evaluar_variantes.py`): una batería de preguntas con respuesta esperada, calificada por un **LLM-juez**. Sirve para medir % de acierto y detectar fallos (no para entrenar). Reveló que con preguntas reformuladas el acierto real era ~66% → con las mejoras subió a ~80%.

---

## 8. Paso E — El pipeline RAG (cómo responde) — `app/rag.py`

Cuando llega una pregunta:
1. **Embedding de la pregunta** (`input_type="query"`).
2. **Búsqueda híbrida en paralelo** (3 fuentes, con `asyncio.gather`):
   - **Vectorial**: función `match_document_chunks` en Supabase (similitud coseno).
   - **Palabra clave**: busca términos exactos importantes (ej. "pilares") y los rankea — cubre lo que la semántica a veces no pesca.
   - **Banco verificado**: similitud coseno solo sobre los ~66 chunks curados (en memoria, rapidísimo); garantiza la respuesta revisada.
3. **Barrera anti-alucinación**: si la mejor similitud es muy baja y no hay banco verificado ni historial → responde "no encontré información" SIN llamar al LLM (evita inventar).
4. **Combinar (merge)**: se arma el contexto **priorizando los chunks verificados**, luego palabra clave, luego vectoriales (máximo 6 fragmentos).
5. **Generar**: se llama al LLM con un **system prompt institucional estricto** (consultor del PMDI, no inventa, cita fuentes, redirige trámites personales) + el contexto + el historial.
6. Devuelve **respuesta + lista de fuentes**.

---

## 9. Paso F — Backend (FastAPI) y streaming — `app/main.py`

Endpoints principales:
- `POST /query` — respuesta completa (JSON).
- `POST /query-stream` — **streaming (SSE)**: envía la respuesta por fragmentos a medida que el LLM la genera (el usuario empieza a leer en segundos). Eventos: `delta`, `done` (con fuentes), `error`.
- `GET /conversation/{id}` — recupera el historial guardado.
- `GET /admin/stats`, `GET /admin/documents` — panel admin.
- `GET /` y `/admin` — sirven el frontend (con cabeceras `no-cache`).

El historial se guarda en Supabase (`chat_messages`) por `conversation_id`.

---

## 10. Paso G — Frontend (chat, memoria, identidad) — `frontend/index.html`

- **HTML/CSS/JS puro** (sin frameworks), con la **identidad visual de la Alcaldía de Medellín** (paleta cian Pantone 306 C, tipografía Arial, escudo oficial extraído del manual de identidad).
- **Memoria conversacional**: genera un `conversation_id` (guardado en `localStorage`), lo envía con cada pregunta junto a los últimos turnos, y al abrir recupera la conversación guardada.
- **Streaming**: consume el SSE y muestra la respuesta token por token; al final renderiza Markdown + el **membrete de fuentes con chips** (las verificadas con chip cian "✓ Respuesta verificada").
- **Móvil-first**: pantalla completa (`100dvh`), input a 16px (evita el zoom de iOS), chips que envuelven.

---

## 11. Paso H — Despliegue en Hostinger (Docker + Nginx Proxy Manager)

1. **Repositorio en GitHub** (público; el `.env` con las claves NO se sube, está en `.gitignore`).
2. **Imagen Docker** (`Dockerfile`): Python 3.11, instala dependencias, copia `app/` y `frontend/`, corre `uvicorn`.
3. **`docker-compose.yml`**: un servicio `api` que publica el puerto `7070:8000` y se conecta a la red Docker existente de Nginx Proxy Manager (`red-de-sitios_red-web`) **sin tocar los otros sitios**.
4. **En el VPS** (por SSH):
   ```bash
   git clone <repo> ~/agente-plan-maestro
   cd ~/agente-plan-maestro
   # crear .env con las claves (NVIDIA + Supabase)
   docker compose up -d --build
   ```
5. **Nginx Proxy Manager** (interfaz web): se creó un **Proxy Host** que enruta el dominio `agentepmdi.centrodepensamientoitm.cloud` → contenedor `agente-plan-maestro` puerto **8000**, con **certificado Let's Encrypt** (HTTPS automático).
6. **DNS**: un registro A del subdominio apuntando al servidor (ya existía para los demás subdominios).

Actualizar en producción = `git pull && docker compose up -d --build`.

---

## 12. Portabilidad: cambiar a Azure / Gemini / DeepSeek

La gran ventaja del diseño: **las piezas son intercambiables**. Aquí el detalle de qué tocar.

### 12.1 Cambiar el LLM (el que redacta) — lo más fácil
Como se usa el cliente OpenAI, casi siempre basta cambiar `base_url`, `api_key` y `model`:

| Proveedor | base_url | model (ejemplo) | Notas |
|---|---|---|---|
| **NVIDIA** (actual) | `https://integrate.api.nvidia.com/v1` | `meta/llama-3.1-8b-instruct` | — |
| **DeepSeek** | `https://api.deepseek.com` | `deepseek-chat` | Compatible OpenAI. Cambio casi directo. |
| **Gemini** | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.0-flash` | Google ofrece endpoint compatible OpenAI. |
| **Azure OpenAI** | (tu endpoint Azure) | nombre de tu *deployment* (ej. `gpt-4o-mini`) | Usa `AzureOpenAI(azure_endpoint=..., api_version=..., api_key=...)` y el `model` es el **deployment name**. |

En el código, esto vive en `app/llm.py` (y `app/embeddings.py`). En la práctica: editar el `.env` (`LLM_MODEL`, `NVIDIA_BASE_URL`→base del nuevo proveedor, `NVIDIA_API_KEY`→nueva key). Para Azure se ajusta el cliente a `AzureOpenAI`.

### 12.2 Cambiar el modelo de EMBEDDINGS — requiere re-ingesta
⚠️ El punto más importante: **cada modelo de embeddings produce vectores de distinta dimensión.** Si lo cambias, debes (1) cambiar `vector(N)` en la tabla y (2) **re-vectorizar todo**.

| Proveedor | Modelo | Dimensión |
|---|---|---|
| **NVIDIA** (actual) | `nv-embedqa-e5-v5` | **1024** |
| **Azure OpenAI** | `text-embedding-3-small` | 1536 |
| **Azure OpenAI** | `text-embedding-3-large` | 3072 |
| **Gemini** | `text-embedding-004` | 768 |
| **DeepSeek** | (no tiene embeddings propios) | usa otro proveedor para embeddings |

Pasos para cambiar embeddings:
1. En `sql/init.sql`, cambiar `embedding vector(1024)` por la nueva dimensión y recrear la tabla.
2. Quitar el `extra_body={"input_type": ...}` (es específico de NVIDIA; Azure/Gemini no lo usan).
3. Volver a correr `scripts/ingest.py` y `scripts/ingest_qa.py` para re-vectorizar con el nuevo modelo.

### 12.3 Cambiar la base de datos vectorial
- **Quedarte en Postgres pero en Azure**: usa **Azure Database for PostgreSQL** con la extensión pgvector → el SQL es casi idéntico; solo cambia la cadena de conexión.
- **Azure AI Search**: servicio de búsqueda vectorial gestionado de Azure; cambia la capa `app/database.py` (en vez de `match_document_chunks`, usas su SDK).
- **Otros**: Pinecone, Qdrant, Weaviate. En todos, lo que cambia es `app/database.py` (cómo se insertan y se buscan los vectores). El resto del pipeline es igual.

### 12.4 Cambiar el hosting
- **Azure Container Apps / App Service for Containers**: usas el **mismo `Dockerfile`**. Subes la imagen al **Azure Container Registry** y despliegas. El HTTPS y el dominio los maneja Azure (en vez de Nginx Proxy Manager).
- El `.env` pasa a ser "variables de entorno"/"secrets" del servicio de Azure.

### 12.5 Resumen de "qué archivos tocar" según el cambio
| Quiero cambiar… | Toco… |
|---|---|
| Solo el LLM (redactor) | `.env` (model/base_url/key); para Azure, `app/llm.py` |
| El modelo de embeddings | `app/embeddings.py`, `sql/init.sql` (dimensión) + **re-ingesta** |
| La base vectorial | `app/database.py` (y el SQL si sigue siendo Postgres) |
| El hosting | `Dockerfile`/`docker-compose.yml` se reusan; cambia el destino del despliegue |

---

## 13. Decisiones y aprendizajes clave (los "tropiezos")

- **La dimensión del vector debe coincidir con el modelo de embeddings.** Cambiar de modelo ⇒ re-ingestar.
- **NVIDIA exige `input_type`** (`passage`/`query`) en embeddings; otros proveedores no.
- **Caracteres nulos `\x00`** en PDFs rompen PostgreSQL: hay que limpiarlos.
- **Acentos en nombres de archivo**: se corrompieron (mojibake) y se corrigieron normalizando a NFC.
- **El `.env` en el VPS** se corrompía al pegar claves largas; solución: base64 o `scp`.
- **Latencia del tier gratuito de NVIDIA**: muy variable (segundos a un minuto), independiente del tamaño del modelo. El **streaming** mejora la percepción; para velocidad garantizada se necesitaría un proveedor de pago.
- **Búsqueda híbrida + banco verificado**: la semántica sola fallaba con términos sobrecargados; combinar con palabra clave y priorizar el banco curado fue lo que más subió la calidad.
- **Evaluar con paráfrasis** (no con las preguntas exactas) revela el rendimiento real (evita el "engaño" del 100%).

---

## 14. Glosario

- **RAG**: técnica para que un LLM responda sobre tus documentos sin re-entrenarlo, dándole contexto recuperado.
- **Embedding**: vector numérico que representa el significado de un texto.
- **Chunk**: fragmento de un documento.
- **pgvector**: extensión de PostgreSQL para guardar y buscar vectores.
- **Similitud coseno**: medida de qué tan "parecidos" son dos vectores (0 a 1).
- **LLM**: modelo de lenguaje grande (el que redacta).
- **SSE (Server-Sent Events)**: forma de enviar la respuesta por partes (streaming) desde el servidor.
- **RLS**: seguridad a nivel de fila en la base de datos.
- **Fine-tuning**: re-entrenar un modelo (NO se hace aquí; se usa RAG).
- **Mojibake**: texto con caracteres corrompidos por una codificación incorrecta.

---

> Documento de referencia del proyecto. Para detalles operativos ver **README.md** (resumen) y **SETUP.md** (instalación/despliegue).
