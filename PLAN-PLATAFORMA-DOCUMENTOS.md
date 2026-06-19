# Plan: Plataforma para subir y gestionar documentos

## Objetivo
Desde el **panel admin** (navegador) subir un documento (PDF o Word) y que el sistema
lo procese y lo agregue al conocimiento del agente — sin terminal ni scripts manuales.

## Flujo
```
Admin sube documento (navegador)
   → Backend recibe el archivo
   → Extrae texto (PyMuPDF para PDF, python-docx para Word)
   → Divide en fragmentos
   → Genera embeddings (NVIDIA)
   → Inserta en Supabase
   → Disponible para consultas al instante
```

## Decisiones acordadas (2026-06-19)
1. **Varias cuentas** podrán subir documentos → autenticación multiusuario (no una sola contraseña).
2. **PDF y Word (.docx)** → se agrega lectura de Word además del PDF que ya existe.
3. **Se podrán eliminar** documentos (para actualizar el contenido).

## Componentes a construir
1. **Autenticación del admin (PRIMERO)** — hoy `/admin` es público; proteger con login
   multiusuario antes de exponer subida/borrado.
2. **Backend — nuevos endpoints**
   - `POST /admin/upload` — recibe el archivo, lo procesa e ingesta.
   - `DELETE /admin/document/{nombre}` — elimina un documento y todos sus fragmentos.
   - `GET /admin/documents` — (ya existe) lista documentos con nº de fragmentos.
3. **Procesamiento en segundo plano** con barra de progreso (vectorizar tarda).
4. **Tabla `documents` en Supabase** — nombre, fecha, nº de fragmentos, estado
   (procesando / listo / error).
5. **Frontend admin mejorado** — botón subir (arrastrar y soltar), barra de progreso,
   botón eliminar por documento, estado de cada uno.
6. **Infraestructura** — subir el límite de tamaño de archivo en Nginx Proxy Manager;
   guardar los documentos originales en un volumen Docker.

## Fases
1. Autenticación multiusuario del admin
2. Subir y procesar un documento (PDF/Word)
3. Eliminar documentos
4. Tabla de control + estados + barra de progreso
5. Pulido: límites de tamaño, volumen para archivos, validaciones

## Notas técnicas
- Reutilizar la lógica de `scripts/ingest.py` (extract_text, split_text,
  generate_embeddings con `input_type=passage`, upsert a Supabase).
- Para Word usar `python-docx` o `docx2txt`.
- Cuidado con `chunk_id`: hoy es un orden global; para subidas incrementales usar
  `max(chunk_id)+1` o un esquema que no colisione.

## Mientras tanto: agregar documentos HOY (manual)
1. Poner los PDFs nuevos en la carpeta `Documentos/`.
2. Ejecutar en la PC:
   ```
   ./venv/Scripts/python.exe scripts/ingest.py
   ```
   (es idempotente — no duplica los que ya están).
