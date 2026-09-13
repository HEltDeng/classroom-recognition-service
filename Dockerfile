FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PADDLEOCR_MODEL_NAME=PP-OCRv5_mobile_rec \
    PADDLEOCR_DEVICE=cpu

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY templates ./templates
COPY docs ./docs

RUN pip install --no-cache-dir . \
    && pip install --no-cache-dir "paddleocr>=3,<4" "paddlepaddle>=3,<4"

EXPOSE 8092
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8092"]
