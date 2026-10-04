# Приложение: OCR, правила, веб-интерфейс и командная строка. Модели работают в отдельном контейнере (см. docker-compose.yml).
FROM python:3.11-slim

# libgl1 и libglib2.0-0 нужны OpenCV, на котором работает OCR
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Модели OCR скачиваются один раз при сборке: контейнеру при работе интернет не нужен
RUN python -c "from courtdocs.ocr import OcrEngine; OcrEngine()._load()"

ENV COURTDOCS_WORKDIR=/data \
    COURTDOCS_LLM_URL=http://ollama:11434 \
    PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8765

CMD ["python", "-m", "uvicorn", "courtdocs.web:app", "--host", "0.0.0.0", "--port", "8765"]
