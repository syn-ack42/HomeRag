import asyncio
import json
import logging
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import HTTPException
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app_core.bot_registry import (
    DEFAULT_BOT_ID,
    DEFAULT_PROMPT,
    bot_paths,
    default_bot_config,
    ensure_bot_exists,
    save_bot_config,
    sanitize_config,
)
from app_core.embedding_engine import EmbeddingPipelineConfig, ParallelEmbeddingEngine
from app_core.index_manager import IndexState, IndexingManager, compute_file_hashes
from app_core.vectorstore_backend import DEFAULT_VECTORSTORE_BACKEND, VectorStoreBackend
from app_core.document_loader import load_documents
from app_services.container import get_service_container


service_container = get_service_container()
settings = service_container.settings

OllamaLLM = Ollama

DEFAULT_EMBEDDING_MODEL = settings.DEFAULT_EMBEDDING_MODEL
DEFAULT_LLM_MODEL = settings.DEFAULT_LLM_MODEL

logger = logging.getLogger("homerag")


def embedding_pipeline_config(config: Dict[str, Any]) -> EmbeddingPipelineConfig:
    return EmbeddingPipelineConfig(
        batch_size=config.get("embedding_batch_size")
        or settings.EMBEDDING_BATCH_SIZE,
        max_concurrency=config.get("embedding_concurrency")
        or settings.EMBEDDING_MAX_WORKERS,
        cache_size=settings.EMBEDDING_CACHE_SIZE,
    )


def duration_ms(start_time: float) -> float:
    return (time.perf_counter() - start_time) * 1000


def log_performance(event: str, start_time: float, **details):
    elapsed = duration_ms(start_time)
    extras = ", ".join(f"{key}={value}" for key, value in details.items())
    suffix = f" ({extras})" if extras else ""
    logger.info("%s completed in %.2f ms%s", event, elapsed, suffix)
    return elapsed


class LoggingOllamaEmbeddings:
    """Wrapper around OllamaEmbeddings with extra logging."""

    def __init__(self, pipeline_config: EmbeddingPipelineConfig, **kwargs):
        self._delegate = OllamaEmbeddings(**kwargs)
        self._model = kwargs.get("model")
        self._base_url = kwargs.get("base_url")
        self._engine = ParallelEmbeddingEngine(self._delegate, pipeline_config)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        start = time.perf_counter()
        logger.info(
            "Embedding %d documents via Ollama (model=%s, url=%s)",
            len(texts),
            self._model,
            self._base_url,
        )
        try:
            vectors = self._engine.embed_documents_sync(texts)
        except Exception:
            logger.exception(
                "Embedding documents failed (model=%s, url=%s)", self._model, self._base_url
            )
            raise

        log_performance(
            "embed_documents",
            start,
            count=len(texts),
            model=self._model,
            base_url=self._base_url,
        )
        return vectors

    def embed_query(self, text: str) -> List[float]:
        start = time.perf_counter()
        logger.info(
            "Embedding query via Ollama (chars=%d, model=%s, url=%s)",
            len(text or ""),
            self._model,
            self._base_url,
        )
        try:
            vector = self._engine.embed_query_sync(text)
        except Exception:
            logger.exception("Embedding query failed (model=%s, url=%s)", self._model, self._base_url)
            raise

        log_performance(
            "embed_query",
            start,
            chars=len(text or ""),
            model=self._model,
            base_url=self._base_url,
        )
        return vector


def embedding_db_path(bot_id: str, embedding_model: str) -> Path:
    return bot_paths(bot_id)["db"] / embedding_model


def get_ollama_url() -> str:
    return str(settings.OLLAMA_URL).rstrip("/")


def fetch_installed_models() -> List[str]:
    base_url = get_ollama_url()
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to fetch models from Ollama: %s", exc)
        raise HTTPException(status_code=503, detail="Unable to load models from Ollama")

    try:
        data = response.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Invalid response from Ollama")

    models = []
    seen = set()
    for model in data.get("models", []):
        name = model.get("model") or model.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        models.append(name)
    return models


def _fetch_model_data() -> List[Dict[str, Any]]:
    base_url = get_ollama_url()

    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to fetch models from Ollama: %s", exc)
        raise HTTPException(status_code=503, detail="Unable to load models from Ollama")

    try:
        data = response.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Invalid response from Ollama")

    return data.get("models", [])


def fetch_installed_embedding_models() -> List[str]:
    embeddings: List[str] = []
    seen = set()
    for model in _fetch_model_data():
        name = model.get("model") or model.get("name")
        if not name or name in seen:
            continue

        details = model.get("details") or {}
        families = details.get("families") or []
        families = [str(fam).lower() for fam in families if fam]
        is_embedding = any(
            fam in {"bert", "nomic-bert", "embed", "bge", "gte"} for fam in families
        )

        if is_embedding:
            embeddings.append(name)
            seen.add(name)

    return embeddings


def fetch_installed_llm_models() -> List[str]:
    llms: List[str] = []
    seen = set()
    for model in _fetch_model_data():
        name = model.get("model") or model.get("name")
        if not name or name in seen:
            continue

        details = model.get("details") or {}
        families = details.get("families") or []
        families = [str(fam).lower() for fam in families if fam]
        is_embedding = any(
            fam in {"bert", "nomic-bert", "embed", "bge", "gte"} for fam in families
        )

        if not is_embedding:
            llms.append(name)
            seen.add(name)

    return llms


def load_bot_config(bot_id: str) -> Dict[str, Any]:
    paths = bot_paths(bot_id)
    config_file = paths.get("config_file")
    if not config_file.exists():
        config = default_bot_config()
        embedding_path = paths.get("embedding_model")
        if embedding_path and embedding_path.exists():
            config["embedding_model"] = embedding_path.read_text().strip()
        save_bot_config(bot_id, config)
        return config

    try:
        raw = json.loads(config_file.read_text())
        return sanitize_config(raw)
    except Exception:
        return default_bot_config()


def current_embedding_model(bot_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> str:
    if config:
        return config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    if bot_id:
        return load_bot_config(bot_id).get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    return DEFAULT_EMBEDDING_MODEL


def current_llm_model(bot_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> str:
    if config:
        return config.get("llm_model", DEFAULT_LLM_MODEL)
    if bot_id:
        return load_bot_config(bot_id).get("llm_model", DEFAULT_LLM_MODEL)
    return DEFAULT_LLM_MODEL


def save_embedding_model(bot_id: str, model_name: str):
    config = load_bot_config(bot_id)
    config["last_built_embedding_model"] = model_name
    save_bot_config(bot_id, config)
    paths = bot_paths(bot_id)
    paths["embedding_model"].write_text(model_name)


def ensure_installed_embedding_model(
    bot_id: str,
    requested_model: str,
    config: Dict[str, Any],
    installed: Optional[List[str]] = None,
) -> Optional[str]:
    installed = installed or fetch_installed_embedding_models()
    if requested_model in installed:
        return requested_model

    replacement = None
    if DEFAULT_EMBEDDING_MODEL in installed:
        replacement = DEFAULT_EMBEDDING_MODEL
    elif installed:
        replacement = installed[0]

    if replacement:
        logger.warning(
            "Embedding model %s not installed — switching to %s",
            requested_model,
            replacement,
        )
        config["embedding_model"] = replacement
        save_bot_config(bot_id, config)
        paths = bot_paths(bot_id)
        paths["embedding_model"].write_text(replacement)
        return replacement

    logger.warning(
        "Embedding model %s not installed and no embedding models are available",
        requested_model,
    )
    return None


def load_saved_embedding_model(bot_id: str) -> Optional[str]:
    config = load_bot_config(bot_id)
    if config.get("last_built_embedding_model"):
        return config.get("last_built_embedding_model")
    path = bot_paths(bot_id)["embedding_model"]
    if path.exists():
        return path.read_text().strip()
    return None


def load_system_prompt(bot_id: str) -> str:
    ensure_bot_exists(bot_id)
    prompt_file = bot_paths(bot_id)["prompt"]
    if prompt_file.exists():
        return prompt_file.read_text()
    save_system_prompt(bot_id, DEFAULT_PROMPT)
    return DEFAULT_PROMPT


def save_system_prompt(bot_id: str, text: str):
    paths = bot_paths(bot_id)
    paths["prompt"].write_text(text or DEFAULT_PROMPT)


def load_prompt_template(bot_id: str, config: Dict[str, Any]) -> str:
    template_path = config.get("prompt_template_path") or ""
    if template_path:
        candidate = Path(template_path)
        if not candidate.is_absolute():
            candidate = bot_paths(bot_id)["config"] / candidate
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved.exists():
            return resolved.read_text()
        logger.warning(
            "Prompt template %s not found for bot %s; falling back to system prompt",
            resolved,
            bot_id,
        )
    return load_system_prompt(bot_id)


def build_db(
    bot_id: str,
    config: Optional[Dict[str, Any]] = None,
    embedding_models: Optional[List[str]] = None,
):
    logger.debug("Starting database rebuild for bot %s", bot_id)
    config = config or load_bot_config(bot_id)
    installed_embeddings = fetch_installed_embedding_models()
    if embedding_models is None:
        embedding_models = [current_embedding_model(bot_id=bot_id, config=config)]
    embedding_models = list(dict.fromkeys(str(m) for m in embedding_models if m))
    resolved_embeddings: List[str] = []
    for embedding_model in embedding_models:
        resolved = ensure_installed_embedding_model(
            bot_id,
            embedding_model,
            config,
            installed_embeddings,
        )
        if resolved and resolved not in resolved_embeddings:
            resolved_embeddings.append(resolved)
    embedding_models = resolved_embeddings
    start_time = time.perf_counter()
    load_start = time.perf_counter()
    docs = load_documents(bot_id)
    file_hashes = compute_file_hashes(bot_id)
    log_performance(
        "load_documents",
        load_start,
        bot_id=bot_id,
        documents=len(docs),
        files=len(file_hashes),
    )
    if not docs:
        db_dir = bot_paths(bot_id)["db"]
        if db_dir.exists():
            shutil.rmtree(db_dir, ignore_errors=True)
        db_dir.mkdir(parents=True, exist_ok=True)
        elapsed_ms = duration_ms(start_time)
        logger.debug("No documents found for bot %s; cleared DB in %.2f ms", bot_id, elapsed_ms)
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.get("chunk_size", 800),
        chunk_overlap=config.get("chunk_overlap", 100),
    )
    split_start = time.perf_counter()
    chunks = splitter.create_documents(
        [d["page_content"] for d in docs],
        metadatas=[d["metadata"] for d in docs]
    )
    chunk_counts: Dict[str, int] = defaultdict(int)
    for doc in chunks:
        source = doc.metadata.get("source") or "unknown"
        idx = chunk_counts[source]
        chunk_counts[source] += 1
        doc.metadata["chunk_id"] = f"{source}::chunk-{idx}"
    log_performance(
        "split_documents",
        split_start,
        bot_id=bot_id,
        documents=len(docs),
        chunks=len(chunks),
    )

    base_url = get_ollama_url()
    backend: VectorStoreBackend = service_container.vectorstore_backend or DEFAULT_VECTORSTORE_BACKEND  # type: ignore[assignment]
    for embedding_model in embedding_models:
        embeddings = LoggingOllamaEmbeddings(
            embedding_pipeline_config(config),
            base_url=base_url,
            model=embedding_model,
        )

        db_dir = embedding_db_path(bot_id, embedding_model)
        manager = IndexingManager(db_dir)
        previous_state = manager.load()
        chunk_changed = (
            previous_state.chunk_size != config.get("chunk_size", 800)
            or previous_state.chunk_overlap != config.get("chunk_overlap", 100)
        )
        db_file = db_dir / "chroma.sqlite3"
        full_rebuild = chunk_changed or not db_file.exists()
        changed_files, removed_files = IndexingManager.diff_hashes(file_hashes, previous_state.file_hashes)

        if full_rebuild:
            if db_dir.exists():
                shutil.rmtree(db_dir)
            db_dir.mkdir(parents=True, exist_ok=True)
            embed_start = time.perf_counter()
            vectordb = backend.build_from_documents(
                chunks,
                embeddings,
                db_dir,
                ids=[doc.metadata.get("chunk_id", "") for doc in chunks],
            )
            backend.persist(vectordb)
            log_performance(
                "build_vectorstore",
                embed_start,
                bot_id=bot_id,
                chunks=len(chunks),
                embedding_model=embedding_model,
            )
        else:
            vectordb = backend.load(db_dir, embeddings)
            if removed_files or changed_files:
                for source in sorted(removed_files | changed_files):
                    backend.delete(vectordb, where={"source": source})
            updated_chunks = [
                doc for doc in chunks if doc.metadata.get("source") in changed_files
            ]
            if updated_chunks:
                embed_start = time.perf_counter()
                backend.add_documents(
                    vectordb,
                    updated_chunks,
                    ids=[doc.metadata.get("chunk_id", "") for doc in updated_chunks],
                )
                log_performance(
                    "update_vectorstore",
                    embed_start,
                    bot_id=bot_id,
                    chunks=len(updated_chunks),
                    embedding_model=embedding_model,
                    removed=len(removed_files),
                )
            backend.persist(vectordb)

        manager.save(
            IndexState(
                chunk_size=config.get("chunk_size", 800),
                chunk_overlap=config.get("chunk_overlap", 100),
                file_hashes=file_hashes,
            )
        )
        save_embedding_model(bot_id, embedding_model)
    total_ms = duration_ms(start_time)
    logger.debug(
        "Finished database rebuild for bot %s with %d documents in %.2f ms",
        bot_id,
        len(docs),
        total_ms,
    )


async def ensure_db_embeddings(bot_id: str):
    logger.debug("Ensuring embeddings for bot %s", bot_id)
    ensure_start = time.perf_counter()
    config = load_bot_config(bot_id)
    installed_embeddings = fetch_installed_embedding_models()
    data_dir = bot_paths(bot_id)["data"]
    has_documents = any(data_dir.iterdir())
    models_to_check: List[str] = []
    current_model = ensure_installed_embedding_model(
        bot_id,
        current_embedding_model(bot_id=bot_id, config=config),
        config,
        installed_embeddings,
    )
    if current_model:
        models_to_check.append(current_model)
    saved_model = load_saved_embedding_model(bot_id)
    if saved_model and saved_model in installed_embeddings and saved_model not in models_to_check:
        models_to_check.append(saved_model)

    missing_models: List[str] = []
    for model_name in models_to_check:
        db_file = embedding_db_path(bot_id, model_name) / "chroma.sqlite3"
        if not db_file.exists() and has_documents:
            missing_models.append(model_name)

    if missing_models:
        await asyncio.to_thread(build_db, bot_id, config, missing_models)
        log_performance(
            "rebuild_vectorstore",
            ensure_start,
            bot_id=bot_id,
            missing_models=missing_models,
            has_documents=has_documents,
        )
    else:
        log_performance(
            "verify_vectorstore",
            ensure_start,
            bot_id=bot_id,
            missing_models=missing_models,
            has_documents=has_documents,
        )


def join_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(
    bot_id: str,
    question: str,
    config: Dict[str, Any],
    history_text: str,
    system_prompt: str,
):
    embedding_model = current_embedding_model(config=config)
    llm_model = current_llm_model(config=config)
    llm_temperature = config.get("llm_temperature")
    llm_top_p = config.get("llm_top_p")
    llm_max_tokens = config.get("llm_max_output_tokens")
    llm_repeat_penalty = config.get("llm_repeat_penalty")
    retrieval_top_k = config.get("retrieval_top_k", 3)
    ollama_url = get_ollama_url()

    embeddings = LoggingOllamaEmbeddings(
        embedding_pipeline_config(config),
        base_url=ollama_url,
        model=embedding_model,
    )
    backend: VectorStoreBackend = service_container.vectorstore_backend or DEFAULT_VECTORSTORE_BACKEND  # type: ignore[assignment]

    def create_vectordb():
        return backend.load(
            persist_directory=embedding_db_path(bot_id, embedding_model),
            embeddings=embeddings,
        )

    def build_chain(vectordb):
        top_k = retrieval_top_k

        def retrieve_with_logging(query: str):
            search_start = time.perf_counter()
            try:
                results = backend.similarity_search_with_score(vectordb, query, k=top_k)
            except Exception:
                logger.exception("Vector search failed for bot %s", bot_id)
                raise

            docs = [doc for doc, _score in results]

            log_performance(
                "vector_search",
                search_start,
                bot_id=bot_id,
                query_chars=len(query or ""),
                hits=len(results),
                top_k=top_k,
            )

            for idx, (doc, score) in enumerate(results, start=1):
                metadata = doc.metadata or {}
                source = metadata.get("source") or metadata.get("file") or "unknown"
                logger.debug(
                    "Result %d for bot %s: score=%.4f, source=%s, metadata=%s, excerpt=%s",
                    idx,
                    bot_id,
                    score,
                    source,
                    metadata,
                    (doc.page_content or "").replace("\n", " ")[:200],
                )

            return docs

        docs = retrieve_with_logging(question)
        context_text = join_docs(docs)
        llm_params = {
            "base_url": ollama_url,
            "model": llm_model,
        }
        if llm_temperature is not None:
            llm_params["temperature"] = llm_temperature
        if llm_top_p is not None:
            llm_params["top_p"] = llm_top_p
        if llm_max_tokens:
            llm_params["num_predict"] = llm_max_tokens
        if llm_repeat_penalty is not None:
            llm_params["repeat_penalty"] = llm_repeat_penalty

        llm = OllamaLLM(**llm_params)

        def format_prompt(inputs: Dict[str, str]) -> str:
            context = inputs.get("context", "")
            question_text = inputs.get("question", "")
            return f"""{system_prompt}

Conversation so far:
{history_text}

Use ONLY the provided context to answer.

Context:
{context}

User: {question_text}
"""

        return (
            RunnableLambda(lambda q: {"context": context_text, "question": q})
            | RunnableLambda(format_prompt)
            | llm
            | StrOutputParser()
        )

    vectordb = create_vectordb()
    rag_chain = build_chain(vectordb)
    return {
        "rag_chain": rag_chain,
        "embedding_model": embedding_model,
        "create_vectordb": create_vectordb,
        "build_chain": build_chain,
    }
