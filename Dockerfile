FROM python:3.11-slim
WORKDIR /app

# CPU-only PyTorch (the default wheel bundles ~2GB of CUDA libraries you don't need on a VPS)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so containers start fast and work offline
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-base
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY app ./app
COPY data ./data
EXPOSE 8000
# Ingest is incremental, so running it on every start is cheap and keeps the index in sync with data/docs.
CMD ["sh", "-c", "python -m app.ingest && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
