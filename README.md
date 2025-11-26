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
- Docker (for the quickest start) or a local Python toolchain
- Access to an Ollama server (default endpoint: `http://localhost:11434`)
- (Recommended) `virtualenv` or `pyenv` for isolation

### Step-by-step environment setup
1. **Install Ollama** (required whether you run the app locally or in Docker):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
   Start the Ollama service and pull at least one chat model and one embedding model:
   ```bash
   ollama pull phi3
   ollama pull mxbai-embed-large
   ```
2. **Local Python setup** (choose this if you want to run without containers):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r app/requirements.txt
   export OLLAMA_URL=http://localhost:11434
   export DATA_PATH=./data
   export DB_PATH=./chroma
   export CONFIG_PATH=./config
   uvicorn main:app --host 0.0.0.0 --port 8090 --app-dir app
   ```
3. **Docker workflow** (recommended for a sealed environment):
   ```bash
   # Build the FastAPI image and start everything with bind mounts for data/config/indexes
   docker compose up --build

   # Stop the stack when you are done
   docker compose down
   ```
   - The container exposes port `8090` by default.
   - Volumes persist configuration, ingested data, and Chroma indexes (`./config`, `./data`, `./chroma` on the host).
   - Point `OLLAMA_URL` at your Ollama instance; the compose file maps `host.docker.internal` so the container can reach a host-side Ollama.

## Usage
### Running the API
- Local: `uvicorn main:app --host 0.0.0.0 --port 8090` from the `app` directory after installing dependencies.
- Docker: `docker compose up` (as above). The FastAPI app is served on `http://localhost:8090`.

**Behind a reverse proxy with a path prefix**
- Set `ROOT_PATH` to the path segment you want to serve under (e.g., `ROOT_PATH=/HomeRag`) and pass the same value to uvicorn’s `--root-path` if you override the launch command.
- Configure your proxy to forward the prefixed location to the app root (e.g., `location /HomeRag/ { proxy_pass http://hephaistos:8090/; proxy_set_header X-Forwarded-Prefix /HomeRag; }`).
- The web UI now autodetects the path prefix and sends API requests relative to it, so the interface works at URLs like `https://docker.franz-renger.de/HomeRag/`.

### Using the web GUI
Open `http://localhost:8090/static/index.html` (or the root if you routed static files there). The right-hand panel manages bots, credentials, and knowledge bases; the left-hand panel is the chat window.

**Creating and securing bots**
- Use **Active Bot** to switch between bots. **Create Bot** opens a prompt for a bot ID; **Delete Bot** removes it and its data.
- Set a password in the **Bot Controls** section. Checking **Hidden** marks the bot as protected; you must unlock with the password before viewing or changing its settings. **Save Password** updates or clears the credential depending on whether you submit an empty field.

**Building and maintaining a knowledge base**
- The **Knowledge Base Files** accordion lets you upload documents (including ZIP archives that are extracted), create or delete folders, move files, and refresh the file tree. Uploads flow into `DATA_PATH/<bot_id>` on disk.
- Click **Rebuild Index** to force an embedding refresh for the active bot (useful after large document changes). The index section also shows buttons to refresh model lists and reload saved settings.

**Tuning request parameters**
- **System Prompt**: edit and save the bot’s base instructions.
- **Retrieval & Model Settings**: choose the embedding and LLM models discovered from Ollama, adjust chunk size/overlap for splitting, set retrieval depth (Top K), toggle history and the number of turns to keep, and tune generation parameters (temperature, top-p, max output tokens, repeat penalty). Saving persists these values under `CONFIG_PATH/bots/<bot_id>` and immediately applies them to new requests.

### What happens when you build a knowledge base
- When you upload files or click **Rebuild Index**, the app loads documents, splits them into chunks, and writes source-aware metadata for each chunk before embedding.【F:app_core/rag_engine.py†L348-L389】
- For each selected embedding model, the service embeds chunks via Ollama, writes them to a Chroma database under `DB_PATH/<bot_id>/<embedding_model>/`, and persists index metadata (hashes, chunking settings) to support incremental updates.【F:app_core/rag_engine.py†L391-L449】
- On startup the app scans existing bots; if documents exist but a matching embedding database is missing for the configured model, it triggers an embedding build in the background so the vector index is ready before the first query.【F:app/main.py†L32-L44】【F:app_core/rag_engine.py†L488-L518】

### Making new models available (autodiscovery)
The service relies on Ollama’s `/api/tags` endpoint to discover installed models at runtime. Both LLM and embedding lists are fetched dynamically, so you only need to install models in Ollama:
```bash
ollama pull mistral
ollama pull mxbai-embed-large
```
Once pulled, the models appear in `/models` and `/embedding-models` API responses and can be assigned to bots without restarting the service.

### Recommended Ollama models by hardware profile
- **General purpose machine (CPU-only or low RAM)**: `phi3` for chat, `nomic-embed-text` for embeddings (small, lightweight pulls).
- **Developer laptop with modest GPU (e.g., 8–12 GB VRAM)**: `mistral` or `llama3` for chat, `mxbai-embed-large` for embeddings.
- **Mid-range gaming rig (e.g., 12–24 GB VRAM)**: `llama3:instruct` or `mixtral` for higher-quality chat, `all-minilm` or `gte-base` variants for embeddings.
- **Specialized LLM hardware (multi-GPU or high-VRAM workstation)**: `llama3:70b` or other large-context models for chat, paired with `nomic-embed-text:v1.5` or similar higher-capacity embedding models.

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
