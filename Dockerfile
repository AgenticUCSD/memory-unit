FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    HOST=0.0.0.0 \
    MEMORY_PERSIST_DIR=/tmp/memory_data

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Cloud Run's reliable writable path is /tmp; per-user Chroma/JSONL state lands under
# MEMORY_PERSIST_DIR (namespaced per user by the app), not the read-only image tree.
RUN mkdir -p /tmp/memory_data

EXPOSE 8080

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080}"]
