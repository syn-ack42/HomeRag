"""Utilities for loading, validating, and manipulating bot documents."""

import shutil
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

import markdown
from bs4 import BeautifulSoup
from fastapi import HTTPException, UploadFile
from pypdf import PdfReader

from app_core.bot_registry import bot_paths


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


def _store_zip_contents(z: zipfile.ZipFile, destination_dir: Path, data_dir: Path) -> List[str]:
    stored: List[str] = []

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

    return stored


async def store_upload(bot_id: str, file: UploadFile, folder: Optional[str] = None):
    destination_dir = resolve_data_path(bot_id, folder)
    destination_dir.mkdir(parents=True, exist_ok=True)
    data_dir = bot_paths(bot_id)["data"].resolve()

    filename = Path(file.filename or "").name

    is_zip = filename.lower().endswith(".zip") or file.content_type == "application/zip"

    if is_zip:
        content = await file.read()
        z = zipfile.ZipFile(BytesIO(content))
        stored = _store_zip_contents(z, destination_dir, data_dir)
        return {"status": "unzipped", "files": stored}

    validate_filename(filename)
    target = destination_dir / filename
    with open(target, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "uploaded", "files": [target.relative_to(data_dir).as_posix()]}
