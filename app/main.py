from fastapi import FastAPI, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pathlib import Path, PurePosixPath
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

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


DATA_PATH = Path(os.environ.get("DATA_PATH", "/data"))
DB_PATH = Path(os.environ.get("DB_PATH", "/chroma"))
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "/config"))

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
    for path in (DATA_PATH, DB_PATH, CONFIG_PATH):
        path.mkdir(parents=True, exist_ok=True)


def bot_paths(bot_id: str) -> Dict[str, Path]:
    """Return useful paths for a given bot identifier."""
    ensure_base_directories()
    bot_config_path = CONFIG_PATH / bot_id
    bot_data_path = DATA_PATH / bot_id
    bot_db_path = DB_PATH / bot_id
    bot_config_path.mkdir(parents=True, exist_ok=True)
    bot_data_path.mkdir(parents=True, exist_ok=True)
    bot_db_path.mkdir(parents=True, exist_ok=True)
    return {
        "config": bot_config_path,
        "data": bot_data_path,
        "db": bot_db_path,
        "prompt": bot_config_path / "prompt.txt",
    }


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
        CONFIG_PATH / bot_id,
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


def build_db(bot_id: str):
    docs = load_documents(bot_id)
    if not docs:
        db_dir = bot_paths(bot_id)["db"]
        if db_dir.exists():
            shutil.rmtree(db_dir)
            db_dir.mkdir(parents=True, exist_ok=True)
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.create_documents(
        [d["page_content"] for d in docs],
        metadatas=[d["metadata"] for d in docs]
    )

    embeddings = OllamaEmbeddings(
        base_url=os.environ.get("OLLAMA_URL", "http://ollama:11434"),
        model="mxbai-embed-large"
    )

    db_dir = bot_paths(bot_id)["db"]

    if db_dir.exists():
        shutil.rmtree(db_dir)
        db_dir.mkdir(parents=True, exist_ok=True)

    vectordb = Chroma.from_documents(
        chunks, embeddings, persist_directory=str(db_dir)
    )

    vectordb.persist()


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def startup_event():
    registry = load_bot_registry()
    for bot in registry["bots"]:
        bot_id = bot["id"]
        paths = bot_paths(bot_id)
        if not paths["prompt"].exists():
            save_system_prompt(bot_id, DEFAULT_PROMPT)
        db_file = paths["db"] / "chroma.sqlite3"
        if not db_file.exists() and any(paths["data"].iterdir()):
            build_db(bot_id)


@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())


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
    build_db(bot_id)
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

    system_prompt = load_system_prompt(bot_id)

    history_text = ""
    for msg in history:
        role = "User" if msg.get("role") == "user" else "Assistant"
        history_text += f"{role}: {msg.get('content', '')}\n"

    require_bot_access(bot_id, password)

    embeddings = OllamaEmbeddings(
        base_url=os.environ.get("OLLAMA_URL", "http://ollama:11434"),
        model="nomic-embed-text"
    )

    vectordb = Chroma(
        persist_directory=str(bot_paths(bot_id)["db"]),
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever()
    llm = Ollama(
        base_url=os.environ.get("OLLAMA_URL", "http://ollama:11434"),
        model=os.environ.get("MODEL", "mistral")
    )

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

    rag_chain = (
        {
            "context": retriever | (lambda docs: join_docs(docs)),
            "question": RunnablePassthrough()
        }
        | format_prompt
        | llm
        | StrOutputParser()
    )

    async def event_stream():
        async for token in rag_chain.astream(question):
            yield json.dumps({"type": "token", "token": token}) + "\n"

    return StreamingResponse(event_stream(), media_type="text/plain")
