# ---------------------------------------------------------------------------
# SignOra inference backend image
#
#   docker build -t signora-backend -f backend/Dockerfile .
#   docker run -p 8000:8000 -v "$PWD/models:/app/models:ro" signora-backend
#
# Built from the repository root so the models/ folder can be copied in. Mount
# models/ as a volume instead if you prefer not to bake ~190 MB into the image.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SIGNORA_HOST=0.0.0.0 \
    SIGNORA_PORT=8000

WORKDIR /app

# System libraries TensorFlow needs at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend/app backend/app
COPY backend/scripts backend/scripts
COPY models /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
