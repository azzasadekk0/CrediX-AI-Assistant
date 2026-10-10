FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CREDIX_RERANKER_DEVICE=cpu

WORKDIR /app
COPY requirements.txt .
# Railway runs this service without a GPU. Install the CPU wheel first so
# sentence-transformers does not pull the multi-gigabyte CUDA distribution.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt
COPY . .

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
