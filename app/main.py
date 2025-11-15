from fastapi import FastAPI, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pathlib import Path
import os
import shutil
import json
import zipfile
from io import BytesIO
from typing import Dict, List, Optional

from pypdf import PdfReader
from bs4 import BeautifulSoup
import markdown

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
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


def load_bot_registry() -> Dict[str, List[Dict[str, str]]]:
    ensure_base_directories()
    if not BOT_REGISTRY_FILE.exists():
        registry = {"bots": [{"id": DEFAULT_BOT_ID, "name": DEFAULT_BOT_NAME}]}
        BOT_REGISTRY_FILE.write_text(json.dumps(registry, indent=2))
        save_system_prompt(DEFAULT_BOT_ID, DEFAULT_PROMPT)
    with BOT_REGISTRY_FILE.open() as f:
        data = json.load(f)
    # Defensive: ensure required keys
    if "bots" not in data or not isinstance(data["bots"], list):
        data = {"bots": [{"id": DEFAULT_BOT_ID, "name": DEFAULT_BOT_NAME}]}
        BOT_REGISTRY_FILE.write_text(json.dumps(data, indent=2))
    return data


def save_bot_registry(registry: Dict[str, List[Dict[str, str]]]):
    ensure_base_directories()
    BOT_REGISTRY_FILE.write_text(json.dumps(registry, indent=2))


def ensure_bot_exists(bot_id: str) -> Dict[str, str]:
    registry = load_bot_registry()
    for bot in registry["bots"]:
        if bot["id"] == bot_id:
            bot_paths(bot_id)  # ensure directories
            return bot
    raise HTTPException(status_code=404, detail="Bot not found")


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


def create_bot(name: str, prompt: Optional[str] = None) -> Dict[str, str]:
    registry = load_bot_registry()
    existing_ids = {bot["id"] for bot in registry["bots"]}
    base_id = slugify(name)
    candidate = base_id
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{base_id}-{suffix}"
        suffix += 1

    registry["bots"].append({"id": candidate, "name": name})
    save_bot_registry(registry)
    save_system_prompt(candidate, prompt or DEFAULT_PROMPT)
    return {"id": candidate, "name": name}


def validate_filename(filename: str) -> str:
    if not filename or filename.strip() == "":
        raise HTTPException(status_code=400, detail="Invalid filename")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return filename


def load_documents(bot_id: str):
    docs = []

    data_dir = bot_paths(bot_id)["data"]

    for file in data_dir.glob("*"):
        if not file.is_file():
            continue

        ext = file.suffix.lower()

        # PDF
        if ext == ".pdf":
            text = extract_text_from_pdf(file)
            docs.append({"page_content": text, "metadata": {"source": file.name}})

        # Text variants
        elif ext in (".txt", ".md"):
            content = file.read_text(errors="ignore")
            if ext == ".md":
                content = extract_text_from_markdown(content)
            docs.append({"page_content": content, "metadata": {"source": file.name}})

        # HTML
        elif ext == ".html":
            html = file.read_text(errors="ignore")
            text = extract_text_from_html(html)
            docs.append({"page_content": text, "metadata": {"source": file.name}})

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
        model="nomic-embed-text"
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
def list_bots() -> Dict[str, List[Dict[str, str]]]:
    registry = load_bot_registry()
    return {"bots": registry["bots"]}


@app.post("/bots")
async def create_bot_endpoint(request: Request):
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required")
    prompt = body.get("prompt")
    bot = create_bot(name, prompt)
    return {"bot": bot}


@app.get("/bots/{bot_id}/files")
def list_files(bot_id: str):
    ensure_bot_exists(bot_id)
    data_dir = bot_paths(bot_id)["data"]
    return {"files": [f.name for f in data_dir.iterdir() if f.is_file()]}


@app.delete("/bots/{bot_id}/file/{filename}")
def delete_file(bot_id: str, filename: str):
    ensure_bot_exists(bot_id)
    validate_filename(filename)
    data_dir = bot_paths(bot_id)["data"]
    target = data_dir / filename
    if target.exists() and target.is_file():
        target.unlink()
        return {"status": "deleted", "file": filename}
    return {"status": "not_found", "file": filename}


@app.post("/bots/{bot_id}/upload")
async def upload(bot_id: str, file: UploadFile):
    ensure_bot_exists(bot_id)
    validate_filename(file.filename)
    target = bot_paths(bot_id)["data"] / file.filename
    with open(target, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "uploaded", "file": file.filename}


@app.post("/bots/{bot_id}/upload_zip")
async def upload_zip(bot_id: str, file: UploadFile):
    ensure_bot_exists(bot_id)
    content = await file.read()
    z = zipfile.ZipFile(BytesIO(content))

    stored = []
    data_dir = bot_paths(bot_id)["data"]

    for name in z.namelist():
        if name.endswith("/"):
            continue
        filename = Path(name).name
        if not filename:
            continue
        validate_filename(filename)
        data = z.read(name)
        target = data_dir / filename
        with open(target, "wb") as f:
            f.write(data)
        stored.append(filename)

    return {"status": "unzipped", "files": stored}


@app.post("/bots/{bot_id}/rebuild")
def rebuild(bot_id: str):
    ensure_bot_exists(bot_id)
    build_db(bot_id)
    return {"status": "reindexed"}


@app.get("/bots/{bot_id}/prompt")
def get_prompt(bot_id: str):
    ensure_bot_exists(bot_id)
    return {"prompt": load_system_prompt(bot_id)}


@app.post("/bots/{bot_id}/prompt")
async def update_prompt(bot_id: str, request: Request):
    ensure_bot_exists(bot_id)
    body = await request.json()
    save_system_prompt(bot_id, body.get("prompt", ""))
    return {"status": "saved"}


def join_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_prompt(bot_id: str):
    system = load_system_prompt(bot_id)
    return ChatPromptTemplate.from_template(
        system + "\n\nContext:\n{context}\n\nQuestion:\n{question}"
    )


@app.post("/ask_stream")
async def ask_stream(request: Request):

    body = await request.json()
    question = body.get("q")
    bot_id = body.get("bot_id") or DEFAULT_BOT_ID

    ensure_bot_exists(bot_id)

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

    prompt = build_prompt(bot_id)

    rag_chain = (
        {
            "context": retriever | (lambda docs: join_docs(docs)),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    async def event_stream():
        async for token in rag_chain.astream(question):
            yield json.dumps({"type": "token", "token": token}) + "\n"

    return StreamingResponse(event_stream(), media_type="text/plain")
