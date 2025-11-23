# HomeRag

HomeRag is a lightweight retrieval-augmented generation (RAG) service built on FastAPI and LangChain. It wraps Ollama-hosted models with document ingestion, vector indexing, and a simple API for asking questions against your own data.

## Features
- FastAPI application with middleware, routes, and static assets bundled together for easy deployment.
- Document ingestion with incremental rebuilds of Chroma vector stores per bot.
- Automatic discovery of available Ollama LLM and embedding models (no manual registry needed).
- Configurable bot profiles including prompts, history, chunking, retrieval depth, and model selection.
- Optional warm-start of embedding and LLM models during startup to reduce first-token latency.

## Installation
### Prerequisites
- Python 3.11+
- Access to an Ollama server (default endpoint: `http://localhost:11434`)
- (Recommended) `virtualenv` or `pyenv` for isolation

### Local setup
1. Create and activate a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies.
   ```bash
   pip install -r app/requirements.txt
   ```
3. Export any environment variables you want to override (see `app_config/settings.py` for defaults), for example:
   ```bash
   export OLLAMA_URL=http://localhost:11434
   export DATA_PATH=./data
   export DB_PATH=./chroma
   export CONFIG_PATH=./config
   ```

### Docker
A ready-to-use image is defined in `app/Dockerfile` and wired in `docker-compose.yml`.
1. Build and start the service:
   ```bash
   docker compose up --build
   ```
2. Volumes persist configuration, ingested data, and Chroma indexes (`./config`, `./data`, `./chroma` on the host). The container exposes port `8090` by default.
3. Set `OLLAMA_URL` to point at your Ollama instance. The compose file maps `host.docker.internal` so the container can reach a host-side Ollama.

## Usage
### Running the API
- Local: `uvicorn main:app --host 0.0.0.0 --port 8090` from the `app` directory after installing dependencies.
- Docker: `docker compose up` (as above). The FastAPI app is served on `http://localhost:8090`.

### Working with bots and documents
- Bot configs live under `CONFIG_PATH/bots/<bot_id>` and are generated on first access. Prompts and embedding model choices are stored alongside the config JSON.
- Upload or place source files under `DATA_PATH/<bot_id>`; embeddings are written to `DB_PATH/<bot_id>/<embedding_model>/` as Chroma databases.
- Rebuilding indexes happens automatically on startup for missing embeddings and when you trigger a rebuild via the API; incremental updates keep unchanged chunks intact.

### Making new models available (autodiscovery)
The service relies on Ollama’s `/api/tags` endpoint to discover installed models at runtime. Both LLM and embedding lists are fetched dynamically, so you only need to install models in Ollama:
```bash
ollama pull mistral
ollama pull mxbai-embed-large
```
Once pulled, the models appear in `/models` and `/embedding-models` API responses and can be assigned to bots without restarting the service.

## Architecture
- **`app/main.py`** bootstraps dependency injection, registers middleware and routers, mounts static assets, and kicks off startup tasks for warming models and rebuilding embeddings when needed.【F:app/main.py†L7-L49】
- **API layer (`app_api/`)** defines middleware and route handlers for bots, files, models, and RAG queries, delegating to core logic while enforcing access controls.【F:app_api/routes/models.py†L1-L35】【F:app_api/middleware.py†L1-L38】
- **Configuration (`app_config/settings.py`)** centralizes environment-driven settings such as paths, defaults, and Ollama endpoint URLs.【F:app_config/settings.py†L1-L27】
- **Core RAG engine (`app_core/`)** manages bot registry persistence, document loading/splitting, vector store updates, and chain construction on top of Ollama and LangChain.【F:app_core/bot_registry.py†L12-L63】【F:app_core/rag_engine.py†L33-L119】
- **Services (`app_services/`)** provide shared infrastructure like the vector store backend (Chroma by default) and a simple service container for dependency sharing.【F:app_services/container.py†L1-L30】【F:app_core/vectorstore_backend.py†L1-L44】
- **Docker (`app/Dockerfile`, `docker-compose.yml`)** packages the app with system dependencies (e.g., `poppler-utils` for PDF parsing) and exposes a single uvicorn process on port 8090 with bind-mounted data/config/index directories.【F:app/Dockerfile†L1-L15】【F:docker-compose.yml†L9-L25】

## Contributing
- Ensure new modules follow the existing layering: API routes call into `app_core`, which uses shared services from `app_services` and configuration from `app_config`.
- Keep environment defaults in `app_config/settings.py` and avoid hard-coding paths in code.
- Format and lint before submitting changes.
