# 🔧 Guía de instalación y despliegue — Asistente Virtual PMDI

## 1. Requisitos
- Cuenta en **NVIDIA build.nvidia.com** (API key gratuita `nvapi-...`)
- Proyecto en **Supabase** (URL + `service_role` key)
- Python 3.11+ (local) · Docker en el VPS

---

## 2. Configurar `.env`
Crea el archivo `.env` en la raíz:
```bash
NVIDIA_API_KEY=nvapi-...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
EMBED_MODEL=nvidia/nv-embedqa-e5-v5
LLM_MODEL=meta/llama-3.1-8b-instruct
SUPABASE_URL=https://TU-PROYECTO.supabase.co
SUPABASE_KEY=eyJ...   # service_role key
```
> En el VPS, si el pegado de la key larga se corta, usa el método base64 (ver memoria del proyecto) o `scp`.

---

## 3. Crear las tablas en Supabase
En **Supabase → SQL Editor**, ejecuta en orden:
1. `sql/init.sql` — extensión `vector`, tabla `document_chunks`, función `match_document_chunks`, RLS.
2. `sql/chat_messages.sql` — tabla `chat_messages` (historial del chat).

---

## 4. Instalar dependencias (local)
```bash
python -m venv venv
venv\Scripts\activate        # Windows (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

---

## 5. Ingesta de datos (una vez, desde local)
```bash
# 1) Vectoriza los 12 PDFs de Documentos/ e inserta en Supabase
python scripts/ingest.py

# 2) Inyecta el banco Q&A verificado (entrenamiento/)
python scripts/ingest_qa.py

# 3) (opcional) "Entrenamiento" extra: variantes parafraseadas
python scripts/generar_variantes.py --n 2
python scripts/aumentar_entrenamiento.py
```
Tras la ingesta, opcionalmente crea el índice vectorial ejecutando `sql/create_index.sql` en Supabase.

---

## 6. Probar en local
```bash
uvicorn app.main:app --reload
```
- Chat: http://localhost:8000/
- Panel admin: http://localhost:8000/admin
- API docs: http://localhost:8000/docs

---

## 7. Desplegar en el VPS (Hostinger)
El proyecto corre como contenedor Docker detrás de **Nginx Proxy Manager** (dominio + HTTPS).

```bash
# En el VPS, por SSH
cd ~/agente-plan-maestro
git pull
docker compose up -d --build
```
- El contenedor expone `agente-plan-maestro:8000` en la red de NPM.
- En NPM, el Proxy Host del dominio apunta a `agente-plan-maestro` puerto `8000`.
- Para cambiar el modelo en producción:
  ```bash
  sed -i 's|^LLM_MODEL=.*|LLM_MODEL=meta/llama-3.1-8b-instruct|' .env
  docker compose up -d --build
  ```

---

## 8. Evaluar la calidad del agente
```bash
# Golden set (12 Q&A + 10 casos + controles) con LLM-juez
python scripts/evaluar_agente.py            # -> entrenamiento/reporte_evaluacion.md

# Generalización con preguntas reformuladas
python scripts/generar_variantes.py --n 2
python scripts/evaluar_variantes.py         # -> entrenamiento/reporte_variantes.md
```

---

## Solución de problemas
- **Sale 0 documentos**: revisa `SUPABASE_KEY` (service_role) y que las tablas existan.
- **Respuesta muy lenta**: es el tier gratuito de NVIDIA (variable). El 8b es más rápido que el 70b. El streaming muestra la respuesta progresivamente.
- **Acentos raros en fuentes**: ya corregido en datos; el frontend además normaliza a NFC.
- **El streaming sale "de golpe" en producción**: en el Proxy Host de NPM → Advanced, añade `proxy_buffering off;`.
