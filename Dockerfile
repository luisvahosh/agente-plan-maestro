FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema para PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar solo lo necesario en runtime (la ingesta ya está hecha en Supabase)
COPY app/ ./app
COPY frontend/ ./frontend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
