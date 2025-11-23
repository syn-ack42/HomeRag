import base64
import hashlib
import hmac
import json
import secrets
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from app_services.container import get_service_container

service_container = get_service_container()
settings = service_container.settings

DATA_PATH = Path(settings.DATA_PATH)
DB_PATH = Path(settings.DB_PATH)
CONFIG_PATH = Path(settings.CONFIG_PATH)
BOT_CONFIG_ROOT = Path(settings.BOT_CONFIG_ROOT)

BOT_REGISTRY_FILE = CONFIG_PATH / "bots.json"
DEFAULT_BOT_ID = "default"
DEFAULT_BOT_NAME = "Default Bot"
PASSWORD_HEADER = "X-Bot-Password"

DEFAULT_PROMPT = """
You are a helpful assistant.
Use ONLY the provided context to answer.
If something is not in the context, say you don't know.
Never hallucinate.
"""

DEFAULT_BOT_CONFIG = {
    "embedding_model": settings.DEFAULT_EMBEDDING_MODEL,
    "llm_model": settings.DEFAULT_LLM_MODEL,
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


def default_bot_config() -> Dict[str, Any]:
    return dict(DEFAULT_BOT_CONFIG)


def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    base = default_bot_config()
    allowed_keys = set(DEFAULT_BOT_CONFIG.keys())
    base.update({k: v for k, v in config.items() if v is not None and k in allowed_keys})

    try:
        base["embedding_model"] = str(base.get("embedding_model") or DEFAULT_BOT_CONFIG["embedding_model"])
        base["llm_model"] = str(base.get("llm_model") or DEFAULT_BOT_CONFIG["llm_model"])
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


def save_bot_config(bot_id: str, config: Dict[str, Any]):
    paths = bot_paths(bot_id)
    sanitized = sanitize_config(config)
    with paths["config_file"].open("w") as f:
        json.dump(sanitized, f, indent=2)


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
        from app.main import save_system_prompt  # local import to avoid circular dependency

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


def password_from_request(request, body: Optional[Dict] = None) -> Optional[str]:
    body = body or {}
    header_password = request.headers.get(PASSWORD_HEADER)
    return body.get("password") or header_password


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
    from app.main import save_system_prompt  # local import to avoid circular dependency

    save_system_prompt(candidate, prompt or DEFAULT_PROMPT)
    save_bot_config(candidate, default_bot_config())
    return {"id": candidate, "name": name, "password_hash": password_hash, "hidden": bool(hidden)}
