from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pathlib import Path
import os
import shutil
import json
import zipfile
from io import BytesIO

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

PROMPT_FILE = CONFIG_PATH / "prompt.txt"

DEFAULT_PROMPT = """
You are a helpful assistant.
Use ONLY the provided context to answer.
If something is not in the context, say you don't know.
Never hallucinate.
"""

def load_system_prompt():
    CONFIG_PATH.mkdir(parents=True, exist_ok=True)
    if PROMPT_FILE.exists():
        return PROMPT_FILE.read_text()
    return DEFAULT_PROMPT

def save_system_prompt(text: str):
    CONFIG_PATH.mkdir(parents=True, exist_ok=True)
    PROMPT_FILE.write_text(text)


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


def load_documents():
    docs = []

    for file in DATA_PATH.glob("*"):
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


def build_db():
    docs = load_documents()
    if not docs:
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

    vectordb = Chroma.from_documents(
        chunks, embeddings, persist_directory=str(DB_PATH)
    )

    vectordb.persist()


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def startup_event():
    if not (DB_PATH / "chroma.sqlite3").exists():
        build_db()


@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())


@app.get("/files")
def list_files():
    return {"files": [f.name for f in DATA_PATH.iterdir() if f.is_file()]}


@app.delete("/file/{filename}")
def delete_file(filename: str):
    target = DATA_PATH / filename
    if target.exists():
        target.unlink()
        return {"status": "deleted", "file": filename}
    return {"status": "not_found", "file": filename}


@app.post("/upload")
async def upload(file: UploadFile):
    target = DATA_PATH / file.filename
    with open(target, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "uploaded"}


@app.post("/upload_zip")
async def upload_zip(file: UploadFile):
    content = await file.read()
    z = zipfile.ZipFile(BytesIO(content))

    for name in z.namelist():
        if name.endswith("/"):
            continue
        data = z.read(name)
        target = DATA_PATH / Path(name).name
        with open(target, "wb") as f:
            f.write(data)

    return {"status": "unzipped", "files": z.namelist()}


@app.post("/rebuild")
def rebuild():
    build_db()
    return {"status": "reindexed"}


@app.get("/prompt")
def get_prompt():
    return {"prompt": load_system_prompt()}


@app.post("/prompt")
async def update_prompt(request: Request):
    body = await request.json()
    save_system_prompt(body.get("prompt", ""))
    return {"status": "saved"}


def join_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_prompt():
    system = load_system_prompt()
    return ChatPromptTemplate.from_template(
        system + "\n\nContext:\n{context}\n\nQuestion:\n{question}"
    )


@app.post("/ask_stream")
async def ask_stream(request: Request):

    body = await request.json()
    question = body.get("q")

    embeddings = OllamaEmbeddings(
        base_url=os.environ.get("OLLAMA_URL", "http://ollama:11434"),
        model="nomic-embed-text"
    )

    vectordb = Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever()
    llm = Ollama(
        base_url=os.environ.get("OLLAMA_URL", "http://ollama:11434"),
        model=os.environ.get("MODEL", "mistral")
    )

    prompt = build_prompt()

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
