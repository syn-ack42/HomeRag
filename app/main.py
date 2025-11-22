import logging
import time

from fastapi import FastAPI, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pathlib import Path, PurePosixPath
import asyncio
import os
import shutil
import json
import zipfile
from io import BytesIO
from typing import Any, Dict, List, Optional
import base64
import hashlib
import hmac
import secrets

from pypdf import PdfReader
from bs4 import BeautifulSoup
import markdown
import requests

from chromadb.errors import InvalidArgumentError

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

# Backwards compatibility alias: older code paths and logs may reference
# `OllamaLLM`, but the current LangChain community package exposes the client as
# `Ollama`. Defining this alias prevents NameError crashes when legacy names are
# used.
OllamaLLM = Ollama

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


DATA_PATH = Path(os.environ.get("DATA_PATH", "/data"))
DB_PATH = Path(os.environ.get("DB_PATH", "/chroma"))
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "/config"))
BOT_CONFIG_ROOT = CONFIG_PATH / "bots"
DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "mxbai-embed-large")
DEFAULT_LLM_MODEL = os.environ.get("MODEL", "phi3")

DEFAULT_PROMPT = """
You are a helpful assistant.
Use ONLY the provided context to answer.
If something is not in the context, say you don't know.
Never hallucinate.
"""

BOT_REGISTRY_FILE = CONFIG_PATH / "bots.json"
DEFAULT_BOT_ID = "default"
DEFAULT_BOT_NAME = "Default Bot"
PASSWORD_HEADER = "X-Bot-Password"


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("homerag")


class LoggingOllamaEmbeddings:
    """Wrapper around OllamaEmbeddings with extra logging.

    This class helps distinguish legitimate embedding calls from runaway loops by
    emitting structured logs whenever embeddings are requested. The interface is
    intentionally minimal and forwards to the underlying LangChain implementation.
    """

    def __init__(self, **kwargs):
        self._delegate = OllamaEmbeddings(**kwargs)
        self._model = kwargs.get("model")
        self._base_url = kwargs.get("base_url")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        start = time.perf_counter()
        logger.info(
            "Embedding %d documents via Ollama (model=%s, url=%s)",
            len(texts),
            self._model,
            self._base_url,
        )
        try:
            vectors = self._delegate.embed_documents(texts)
        except Exception:
            logger.exception("Embedding documents failed (model=%s, url=%s)", self._model, self._base_url)
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
            vector = self._delegate.embed_query(text)
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

DEFAULT_BOT_CONFIG = {
    "embedding_model": DEFAULT_EMBEDDING_MODEL,
    "llm_model": DEFAULT_LLM_MODEL,
    "chunk_size": 800,
    "chunk_overlap": 100,
    "retrieval_top_k": 3,
    "history_enabled": True,
    "history_turns": 10,
    "llm_temperature": 0.0,
    "llm_top_p": 0.9,
    "llm_max_output_tokens": 512,
    "llm_repeat_penalty": 1.1,
    "prompt_template_path": "",
    "last_built_embedding_model": None,
}


def duration_ms(start_time: float) -> float:
    return (time.perf_counter() - start_time) * 1000


def log_performance(event: str, start_time: float, **details):
    elapsed = duration_ms(start_time)
    extras = ", ".join(f"{key}={value}" for key, value in details.items())
    suffix = f" ({extras})" if extras else ""
    logger.info("%s completed in %.2f ms%s", event, elapsed, suffix)
    return elapsed


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100_000)
    return base64.b64encode(salt + derived).decode()


def verify_password(stored_hash: Optional[str], candidate: Optional[str]) -> bool:
    if not stored_hash:
        return False
    if candidate is None:
        return False
    try:
        raw = base64.b64decode(stored_hash)
    except Exception:
        return False
    salt, expected = raw[:16], raw[16:]
    attempt = hashlib.pbkdf2_hmac("sha256", candidate.encode(), salt, 100_000)
    return hmac.compare_digest(expected, attempt)


def is_bot_protected(bot: Dict[str, Any]) -> bool:
    return bool(bot.get("password_hash"))


def ensure_base_directories():
    """Make sure top-level storage directories exist."""
    for path in (DATA_PATH, DB_PATH, CONFIG_PATH, BOT_CONFIG_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def bot_paths(bot_id: str) -> Dict[str, Path]:
    """Return useful paths for a given bot identifier."""
    ensure_base_directories()
    bot_config_path = BOT_CONFIG_ROOT / bot_id
    bot_data_path = DATA_PATH / bot_id
    bot_db_path = DB_PATH / bot_id
    bot_config_path.mkdir(parents=True, exist_ok=True)
    bot_data_path.mkdir(parents=True, exist_ok=True)
    bot_db_path.mkdir(parents=True, exist_ok=True)
    return {
        "config": bot_config_path,
        "config_file": bot_config_path / "config.json",
        "data": bot_data_path,
        "db": bot_db_path,
        "prompt": bot_config_path / "prompt.txt",
        "embedding_model": bot_config_path / "embedding_model.txt",
    }


def embedding_db_path(bot_id: str, embedding_model: str) -> Path:
    return bot_paths(bot_id)["db"] / embedding_model


def default_bot_config() -> Dict[str, Any]:
    return dict(DEFAULT_BOT_CONFIG)


def get_ollama_url() -> str:
    """Return the configured Ollama base URL without a trailing slash."""

    return os.environ.get("OLLAMA_URL", "http://ollama:11434").rstrip("/")


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


def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    base = default_bot_config()
    allowed_keys = set(DEFAULT_BOT_CONFIG.keys())
    base.update({k: v for k, v in config.items() if v is not None and k in allowed_keys})

    try:
        base["embedding_model"] = str(base.get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
        base["llm_model"] = str(base.get("llm_model") or DEFAULT_LLM_MODEL)
        base["chunk_size"] = max(1, int(base["chunk_size"]))
        base["chunk_overlap"] = max(0, int(base["chunk_overlap"]))
        if base["chunk_overlap"] >= base["chunk_size"]:
            base["chunk_overlap"] = max(0, base["chunk_size"] - 1)
        base["retrieval_top_k"] = max(1, int(base["retrieval_top_k"]))
        base["history_enabled"] = bool(base["history_enabled"])
        base["history_turns"] = max(0, int(base.get("history_turns", 0)))
        base["llm_temperature"] = float(base.get("llm_temperature", 0.0))
        base["llm_top_p"] = float(base.get("llm_top_p", 1.0))
        max_tokens = int(base.get("llm_max_output_tokens", 0))
        base["llm_max_output_tokens"] = max_tokens if max_tokens > 0 else None
        base["llm_repeat_penalty"] = float(base.get("llm_repeat_penalty", 1.0))
        base["prompt_template_path"] = str(base.get("prompt_template_path") or "")
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid configuration values")

    return base


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


def save_bot_config(bot_id: str, config: Dict[str, Any]):
    paths = bot_paths(bot_id)
    sanitized = sanitize_config(config)
    with paths["config_file"].open("w") as f:
        json.dump(sanitized, f, indent=2)


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


def normalize_bot(bot: Dict[str, str]) -> Dict[str, str]:
    bot.setdefault("password_hash", None)
    bot.setdefault("hidden", False)
    if "password" in bot:
        if bot["password"]:
            bot["password_hash"] = hash_password(bot["password"])
        bot.pop("password", None)
    return bot


def load_bot_registry() -> Dict[str, List[Dict[str, str]]]:
    ensure_base_directories()
    if not BOT_REGISTRY_FILE.exists():
        registry = {"bots": [normalize_bot({"id": DEFAULT_BOT_ID, "name": DEFAULT_BOT_NAME})]}
        BOT_REGISTRY_FILE.write_text(json.dumps(registry, indent=2))
        save_system_prompt(DEFAULT_BOT_ID, DEFAULT_PROMPT)
        save_bot_config(DEFAULT_BOT_ID, default_bot_config())
    with BOT_REGISTRY_FILE.open() as f:
        data = json.load(f)
    # Defensive: ensure required keys
    if "bots" not in data or not isinstance(data["bots"], list):
        data = {"bots": [normalize_bot({"id": DEFAULT_BOT_ID, "name": DEFAULT_BOT_NAME})]}
        BOT_REGISTRY_FILE.write_text(json.dumps(data, indent=2))
    normalized_bots = []
    changed = False
    for bot in data.get("bots", []):
        original = dict(bot)
        normalized = normalize_bot(bot)
        normalized_bots.append(normalized)
        if normalized != original:
            changed = True
    data["bots"] = normalized_bots
    if changed:
        save_bot_registry(data)
    return data


def save_bot_registry(registry: Dict[str, List[Dict[str, str]]]):
    ensure_base_directories()
    BOT_REGISTRY_FILE.write_text(json.dumps(registry, indent=2))


def get_bot(bot_id: str) -> Dict[str, str]:
    registry = load_bot_registry()
    for bot in registry["bots"]:
        if bot["id"] == bot_id:
            return bot
    raise HTTPException(status_code=404, detail="Bot not found")


def ensure_bot_exists(bot_id: str) -> Dict[str, str]:
    bot = get_bot(bot_id)
    bot_paths(bot_id)
    return bot


def require_bot_access(bot_id: str, password: Optional[str]) -> Dict[str, str]:
    bot = ensure_bot_exists(bot_id)
    stored_hash = bot.get("password_hash")
    if stored_hash and not verify_password(stored_hash, password or ""):
        raise HTTPException(status_code=403, detail="Invalid password for this bot")
    return bot


def update_bot_password(bot_id: str, new_password: Optional[str]):
    registry = load_bot_registry()
    updated = False
    for bot in registry["bots"]:
        if bot["id"] == bot_id:
            bot["password_hash"] = hash_password(new_password) if new_password else None
            updated = True
            break
    if not updated:
        raise HTTPException(status_code=404, detail="Bot not found")
    save_bot_registry(registry)


def remove_bot(bot_id: str):
    if bot_id == DEFAULT_BOT_ID:
        raise HTTPException(status_code=400, detail="Cannot delete the default bot")

    registry = load_bot_registry()
    before = len(registry["bots"])
    registry["bots"] = [bot for bot in registry["bots"] if bot["id"] != bot_id]

    if len(registry["bots"]) == before:
        raise HTTPException(status_code=404, detail="Bot not found")

    save_bot_registry(registry)

    # Clean up storage for the bot if it exists
    for path in (
        BOT_CONFIG_ROOT / bot_id,
        DATA_PATH / bot_id,
        DB_PATH / bot_id,
    ):
        if path.exists():
            shutil.rmtree(path)


def password_from_request(request: Request, body: Optional[Dict] = None) -> Optional[str]:
    body = body or {}
    header_password = request.headers.get(PASSWORD_HEADER)
    return body.get("password") or header_password


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


def extract_text_from_pdf(pdf_path: Path) -> str:
    text = ""
    reader = PdfReader(str(pdf_path))
    for p in reader.pages:
        txt = p.extract_text() or ""
        text += txt + "\n"
    return text


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


def extract_text_from_markdown(md: str) -> str:
    html = markdown.markdown(md)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


def slugify(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in name)
    cleaned = cleaned.strip("-") or "bot"
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned


def create_bot(
    name: str,
    prompt: Optional[str] = None,
    password: Optional[str] = None,
    hidden: bool = False,
) -> Dict[str, str]:
    registry = load_bot_registry()
    existing_ids = {bot["id"] for bot in registry["bots"]}
    base_id = slugify(name)
    candidate = base_id
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{base_id}-{suffix}"
        suffix += 1

    password_hash = hash_password(password) if password else None
    registry["bots"].append(
        normalize_bot(
            {
                "id": candidate,
                "name": name,
                "password_hash": password_hash,
                "hidden": bool(hidden),
            }
        )
    )
    save_bot_registry(registry)
    save_bot_config(candidate, default_bot_config())
    save_system_prompt(candidate, prompt or DEFAULT_PROMPT)
    return {"id": candidate, "name": name, "password_hash": password_hash, "hidden": bool(hidden)}


def validate_filename(filename: str) -> str:
    if not filename or filename.strip() == "":
        raise HTTPException(status_code=400, detail="Invalid filename")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return filename


def sanitize_relative_path(path_value: Optional[str]) -> Path:
    """Return a safe relative path inside a bot's data directory."""

    rel = PurePosixPath(path_value or ".")
    if rel.is_absolute():
        raise HTTPException(status_code=400, detail="Path must be relative")
    if any(part == ".." for part in rel.parts):
        raise HTTPException(status_code=400, detail="Invalid path traversal")

    cleaned_parts = [p for p in rel.parts if p not in ("", ".")]
    return Path(*cleaned_parts)


def resolve_data_path(bot_id: str, relative: Optional[str]) -> Path:
    base = bot_paths(bot_id)["data"].resolve()
    rel_path = sanitize_relative_path(relative)
    target = (base / rel_path).resolve()
    if not target.is_relative_to(base):
        raise HTTPException(status_code=400, detail="Path outside bot data directory")
    return target


def build_tree(base: Path, rel: Path = Path(".")) -> Dict:
    current = base / rel
    node = {
        "name": current.name if rel != Path(".") else "root",
        "path": rel.as_posix() if rel != Path(".") else "",
        "type": "folder",
        "folders": [],
        "files": [],
    }

    if not current.exists():
        return node

    for item in sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if item.is_dir():
            node["folders"].append(build_tree(base, rel / item.name))
        elif item.is_file():
            node["files"].append(
                {
                    "name": item.name,
                    "path": (rel / item.name).as_posix() if rel != Path(".") else item.name,
                    "type": "file",
                }
            )

    return node


def load_documents(bot_id: str):
    docs = []

    data_dir = bot_paths(bot_id)["data"]

    for file in data_dir.rglob("*"):
        if not file.is_file():
            continue

        ext = file.suffix.lower()
        rel_name = file.relative_to(data_dir).as_posix()

        # PDF
        if ext == ".pdf":
            text = extract_text_from_pdf(file)
            docs.append({"page_content": text, "metadata": {"source": rel_name}})

        # Text variants
        elif ext in (".txt", ".md"):
            content = file.read_text(errors="ignore")
            if ext == ".md":
                content = extract_text_from_markdown(content)
            docs.append({"page_content": content, "metadata": {"source": rel_name}})

        # HTML
        elif ext == ".html":
            html = file.read_text(errors="ignore")
            text = extract_text_from_html(html)
            docs.append({"page_content": text, "metadata": {"source": rel_name}})

    return docs


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
    log_performance("load_documents", load_start, bot_id=bot_id, documents=len(docs))
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
    log_performance(
        "split_documents",
        split_start,
        bot_id=bot_id,
        documents=len(docs),
        chunks=len(chunks),
    )

    base_url = get_ollama_url()
    for embedding_model in embedding_models:
        embeddings = LoggingOllamaEmbeddings(
            base_url=base_url,
            model=embedding_model
        )

        db_dir = embedding_db_path(bot_id, embedding_model)
        if db_dir.exists():
            shutil.rmtree(db_dir)
        db_dir.mkdir(parents=True, exist_ok=True)

        embed_start = time.perf_counter()
        vectordb = Chroma.from_documents(
            chunks, embeddings, persist_directory=str(db_dir)
        )
        log_performance(
            "build_vectorstore",
            embed_start,
            bot_id=bot_id,
            chunks=len(chunks),
            embedding_model=embedding_model,
        )

        vectordb.persist()
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
    embedding_model = ensure_installed_embedding_model(
        bot_id,
        current_embedding_model(bot_id=bot_id, config=config),
        config,
        installed_embeddings,
    )
    if not embedding_model:
        log_performance(
            "verify_vectorstore",
            ensure_start,
            bot_id=bot_id,
            missing_models=[],
            has_documents=has_documents,
        )
        return
    db_file = embedding_db_path(bot_id, embedding_model) / "chroma.sqlite3"

    if not db_file.exists() and has_documents:
        await asyncio.to_thread(build_db, bot_id, config, [embedding_model])
        log_performance(
            "rebuild_vectorstore",
            ensure_start,
            bot_id=bot_id,
            missing_models=[embedding_model],
            has_documents=has_documents,
        )
    else:
        log_performance(
            "verify_vectorstore",
            ensure_start,
            bot_id=bot_id,
            missing_models=[] if db_file.exists() else [embedding_model],
            has_documents=has_documents,
        )


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()

    body_bytes: bytes = b""
    try:
        body_bytes = await request.body()
        if body_bytes:
            max_len = 2000
            clipped = body_bytes[:max_len]
            suffix = " ...<truncated>" if len(body_bytes) > max_len else ""
            body_preview = clipped.decode("utf-8", errors="replace")
            logger.debug(
                "Received %s %s with body (%d bytes): %s%s",
                request.method,
                request.url.path,
                len(body_bytes),
                body_preview,
                suffix,
            )
        else:
            logger.debug("Received %s %s with empty body", request.method, request.url.path)
    except Exception as exc:  # pragma: no cover - defensive logging only
        logger.debug(
            "Received %s %s (failed to read body: %s)",
            request.method,
            request.url.path,
            exc,
        )

    async def receive() -> dict:
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    request_with_body = Request(request.scope, receive=receive)

    try:
        response = await call_next(request_with_body)
    except Exception as exc:
        logger.exception("Request %s %s failed: %s", request.method, request.url.path, exc)
        raise

    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.debug(
        "Handled %s %s in %.2f ms with status %s",
        request.method,
        request.url.path,
        duration_ms,
        response.status_code,
    )
    response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"
    return response


@app.on_event("startup")
async def startup_event():
    registry = load_bot_registry()
    app.state.startup_tasks = []
    for bot in registry["bots"]:
        bot_id = bot["id"]
        paths = bot_paths(bot_id)
        load_bot_config(bot_id)
        if not paths["prompt"].exists():
            save_system_prompt(bot_id, DEFAULT_PROMPT)
        if any(paths["data"].iterdir()):
            task = asyncio.create_task(ensure_db_embeddings(bot_id))
            app.state.startup_tasks.append(task)


@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())


@app.get("/models")
def list_models():
    return {"models": fetch_installed_llm_models()}


@app.get("/embedding-models")
def list_embedding_models():
    return {"models": fetch_installed_embedding_models()}


@app.get("/bots")
def list_bots() -> Dict[str, List[Dict[str, Any]]]:
    registry = load_bot_registry()
    return {
        "bots": [
            {
                "id": bot["id"],
                "name": bot["name"],
                "protected": is_bot_protected(bot),
                "hidden": bool(bot.get("hidden")),
            }
            for bot in registry["bots"]
        ]
    }


@app.post("/bots")
async def create_bot_endpoint(request: Request):
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    prompt = body.get("prompt")
    password = (body.get("password") or "").strip() or None
    hidden = bool(body.get("hidden", False))
    bot = create_bot(name, prompt, password, hidden)
    return {
        "bot": {
            "id": bot["id"],
            "name": bot["name"],
            "protected": is_bot_protected(bot),
            "hidden": hidden,
        }
    }


@app.post("/bots/{bot_id}/access")
async def access_bot(bot_id: str, request: Request):
    body = await request.json()
    require_bot_access(bot_id, password_from_request(request, body))
    bot = get_bot(bot_id)
    return {"status": "ok", "protected": is_bot_protected(bot)}


@app.post("/bots/{bot_id}/password")
async def change_bot_password(bot_id: str, request: Request):
    body = await request.json()
    current_password = password_from_request(request, body)
    bot = get_bot(bot_id)
    if is_bot_protected(bot) and not verify_password(bot.get("password_hash"), current_password or ""):
        raise HTTPException(status_code=403, detail="Invalid password for this bot")

    new_password_provided = "new_password" in body
    new_password = (body.get("new_password") or "").strip() if new_password_provided else None
    hidden = bool(body.get("hidden", bot.get("hidden", False)))
    if new_password_provided:
        update_bot_password(bot_id, new_password or None)
    registry = load_bot_registry()
    for b in registry["bots"]:
        if b["id"] == bot_id:
            b["hidden"] = hidden
            break
    save_bot_registry(registry)
    updated_bot = get_bot(bot_id)
    return {"status": "updated", "protected": is_bot_protected(updated_bot), "hidden": hidden}


@app.delete("/bots/{bot_id}")
async def delete_bot(bot_id: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    require_bot_access(bot_id, password_from_request(request, body))
    remove_bot(bot_id)
    return {"status": "deleted"}


@app.get("/bots/{bot_id}/files")
def list_files(bot_id: str, request: Request):
    require_bot_access(bot_id, password_from_request(request))
    data_dir = bot_paths(bot_id)["data"]
    tree = build_tree(data_dir)
    return {"tree": tree}


@app.delete("/bots/{bot_id}/file/{file_path:path}")
def delete_file(bot_id: str, file_path: str, request: Request):
    require_bot_access(bot_id, password_from_request(request))
    target = resolve_data_path(bot_id, file_path)
    data_dir = bot_paths(bot_id)["data"].resolve()
    if target == data_dir:
        raise HTTPException(status_code=400, detail="Cannot delete root directory")
    if target.exists() and target.is_file():
        target.unlink()
        return {"status": "deleted", "file": target.relative_to(data_dir).as_posix()}
    return {"status": "not_found", "file": file_path}


async def store_upload(bot_id: str, file: UploadFile, folder: Optional[str] = None):
    destination_dir = resolve_data_path(bot_id, folder)
    destination_dir.mkdir(parents=True, exist_ok=True)
    data_dir = bot_paths(bot_id)["data"].resolve()

    filename = Path(file.filename or "").name

    is_zip = filename.lower().endswith(".zip") or file.content_type == "application/zip"

    if is_zip:
        content = await file.read()
        z = zipfile.ZipFile(BytesIO(content))
        stored = []

        for info in z.infolist():
            member_path = PurePosixPath(info.filename)
            if info.is_dir():
                target_dir = destination_dir / sanitize_relative_path(member_path.as_posix())
                target_dir.mkdir(parents=True, exist_ok=True)
                continue

            safe_member = sanitize_relative_path(member_path.as_posix())
            if safe_member == Path("."):
                continue

            target = destination_dir / safe_member
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as f:
                f.write(z.read(info.filename))
            stored.append(target.relative_to(data_dir).as_posix())

        return {"status": "unzipped", "files": stored}

    validate_filename(filename)
    target = destination_dir / filename
    with open(target, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "uploaded", "files": [target.relative_to(data_dir).as_posix()]}


@app.post("/bots/{bot_id}/upload")
async def upload(bot_id: str, request: Request, file: UploadFile, path: Optional[str] = None):
    require_bot_access(bot_id, password_from_request(request))
    return await store_upload(bot_id, file, path)


@app.post("/bots/{bot_id}/upload_zip")
async def upload_zip(bot_id: str, request: Request, file: UploadFile, path: Optional[str] = None):
    require_bot_access(bot_id, password_from_request(request))
    return await store_upload(bot_id, file, path)


@app.post("/bots/{bot_id}/folders")
async def create_folder(bot_id: str, request: Request):
    body = await request.json()
    require_bot_access(bot_id, password_from_request(request, body))
    path = body.get("path", "")
    target = resolve_data_path(bot_id, path)
    target.mkdir(parents=True, exist_ok=True)
    rel = target.relative_to(bot_paths(bot_id)["data"].resolve()).as_posix()
    return {"status": "created", "folder": rel}


@app.delete("/bots/{bot_id}/folders")
def delete_folder(bot_id: str, path: str, request: Request):
    require_bot_access(bot_id, password_from_request(request))
    if not path:
        raise HTTPException(status_code=400, detail="Folder path is required")
    target = resolve_data_path(bot_id, path)
    data_dir = bot_paths(bot_id)["data"].resolve()
    if target == data_dir:
        raise HTTPException(status_code=400, detail="Cannot delete root folder")
    if not target.exists():
        return {"status": "not_found", "folder": path}
    shutil.rmtree(target)
    return {"status": "deleted", "folder": target.relative_to(data_dir).as_posix()}


@app.post("/bots/{bot_id}/move_file")
async def move_file(bot_id: str, request: Request):
    body = await request.json()
    require_bot_access(bot_id, password_from_request(request, body))
    source = body.get("source")
    destination_folder = body.get("destination_folder", "")

    if not source:
        raise HTTPException(status_code=400, detail="Source file is required")

    source_path = resolve_data_path(bot_id, source)
    if not source_path.exists() or not source_path.is_file():
        raise HTTPException(status_code=404, detail="Source file not found")

    dest_dir = resolve_data_path(bot_id, destination_folder)
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / source_path.name

    shutil.move(str(source_path), dest_path)

    data_dir = bot_paths(bot_id)["data"].resolve()
    return {
        "status": "moved",
        "from": source_path.relative_to(data_dir).as_posix(),
        "to": dest_path.relative_to(data_dir).as_posix(),
    }


@app.post("/bots/{bot_id}/rebuild")
def rebuild(bot_id: str, request: Request):
    require_bot_access(bot_id, password_from_request(request))
    try:
        build_db(bot_id)
    except Exception as exc:
        logger.exception("Rebuild failed for bot %s", bot_id)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "reindexed"}


@app.get("/bots/{bot_id}/prompt")
def get_prompt(bot_id: str, request: Request):
    require_bot_access(bot_id, password_from_request(request))
    return {"prompt": load_system_prompt(bot_id)}


@app.post("/bots/{bot_id}/prompt")
async def update_prompt(bot_id: str, request: Request):
    body = await request.json()
    require_bot_access(bot_id, password_from_request(request, body))
    save_system_prompt(bot_id, body.get("prompt", ""))
    return {"status": "saved"}


@app.get("/bots/{bot_id}/model")
def get_bot_model(bot_id: str, request: Request):
    require_bot_access(bot_id, password_from_request(request))
    return {"model": current_llm_model(bot_id)}


@app.post("/bots/{bot_id}/model")
async def update_bot_model(bot_id: str, request: Request):
    body = await request.json()
    require_bot_access(bot_id, password_from_request(request, body))
    selected_model = (body.get("model") or "").strip()
    if not selected_model:
        raise HTTPException(status_code=400, detail="Model is required")

    available_models = fetch_installed_llm_models()
    if selected_model not in available_models:
        raise HTTPException(status_code=400, detail="Unknown or unavailable model")

    config = load_bot_config(bot_id)
    config["llm_model"] = selected_model
    save_bot_config(bot_id, config)
    return {"status": "saved", "model": selected_model}


@app.get("/bots/{bot_id}/config")
def get_bot_config(bot_id: str, request: Request):
    require_bot_access(bot_id, password_from_request(request))
    return {"config": load_bot_config(bot_id)}


@app.post("/bots/{bot_id}/config")
async def update_bot_config_endpoint(bot_id: str, request: Request):
    body = await request.json()
    require_bot_access(bot_id, password_from_request(request, body))
    current = load_bot_config(bot_id)
    current.update(body or {})
    save_bot_config(bot_id, current)
    return {"status": "saved", "config": load_bot_config(bot_id)}


def join_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


@app.post("/ask_stream")
async def ask_stream(request: Request):

    body = await request.json()
    question = body.get("question") or body.get("q")
    bot_id = body.get("bot") or body.get("bot_id") or DEFAULT_BOT_ID
    history = body.get("history") or []
    password = password_from_request(request, body)

    if not isinstance(history, list):
        history = []

    bot_config = load_bot_config(bot_id)
    system_prompt = load_prompt_template(bot_id, bot_config)

    embedding_model = current_embedding_model(config=bot_config)
    llm_model = current_llm_model(config=bot_config)
    llm_temperature = bot_config.get("llm_temperature")
    llm_top_p = bot_config.get("llm_top_p")
    llm_max_tokens = bot_config.get("llm_max_output_tokens")
    llm_repeat_penalty = bot_config.get("llm_repeat_penalty")
    retrieval_top_k = bot_config.get("retrieval_top_k", 3)
    history_enabled = bot_config.get("history_enabled", True)
    history_turns = bot_config.get("history_turns", 0)
    ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")

    chat_history = history if history_enabled else []
    max_turns = history_turns
    if max_turns:
        chat_history = chat_history[-max_turns:]

    history_text = ""
    for msg in chat_history:
        role = "User" if msg.get("role") == "user" else "Assistant"
        history_text += f"{role}: {msg.get('content', '')}\n"

    require_bot_access(bot_id, password)
    logger.info(
        "Received question for bot %s (history=%d, question_chars=%d, embedding_model=%s, llm_model=%s, temperature=%s, top_p=%s, max_tokens=%s, repeat_penalty=%s, retrieval_top_k=%s, history_enabled=%s, history_turns=%s, ollama_url=%s)",
        bot_id,
        len(history),
        len(question or ""),
        embedding_model,
        llm_model,
        llm_temperature,
        llm_top_p,
        llm_max_tokens,
        llm_repeat_penalty,
        retrieval_top_k,
        history_enabled,
        history_turns,
        ollama_url,
    )

    ensure_start = time.perf_counter()
    await ensure_db_embeddings(bot_id)
    log_performance("ensure_embeddings", ensure_start, bot_id=bot_id)

    embeddings = LoggingOllamaEmbeddings(
        base_url=ollama_url,
        model=embedding_model,
    )

    def create_vectordb():
        return Chroma(
            persist_directory=str(embedding_db_path(bot_id, embedding_model)),
            embedding_function=embeddings,
        )

    def build_chain(vectordb: Chroma):
        top_k = retrieval_top_k

        def retrieve_with_logging(query: str):
            search_start = time.perf_counter()
            try:
                results = vectordb.similarity_search_with_score(query, k=top_k)
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

    async def event_stream():
        stream_start = time.perf_counter()
        tokens_emitted = 0
        try:
            async for token in rag_chain.astream(question):
                tokens_emitted += 1
                yield json.dumps({"type": "token", "token": token}) + "\n"
        except InvalidArgumentError:
            await asyncio.to_thread(build_db, bot_id, bot_config, [embedding_model])
            rebuilt_vectordb = create_vectordb()
            rebuilt_chain = build_chain(rebuilt_vectordb)
            async for token in rebuilt_chain.astream(question):
                tokens_emitted += 1
                yield json.dumps({"type": "token", "token": token}) + "\n"
        finally:
            log_performance(
                "stream_response",
                stream_start,
                bot_id=bot_id,
                tokens=tokens_emitted,
                question_chars=len(question or ""),
            )

    return StreamingResponse(event_stream(), media_type="text/plain")
