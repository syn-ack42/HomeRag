FROM python:3.11-slim

WORKDIR /app
COPY app/requirements.txt ./requirements.txt
RUN apt-get update && apt-get install -y poppler-utils && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

COPY app/ ./
COPY app_api ./app_api
COPY app_config ./app_config
COPY app_core ./app_core
COPY app_services ./app_services

EXPOSE 8090
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8090"]
