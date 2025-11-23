"""File and folder management routes."""
import shutil
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile

from app_core.bot_registry import bot_paths, password_from_request, require_bot_access
from app_core.document_loader import build_tree, resolve_data_path, store_upload

router = APIRouter()


@router.get("/bots/{bot_id}/files")
def list_files(bot_id: str, request: Request):
    require_bot_access(bot_id, password_from_request(request))
    data_dir = bot_paths(bot_id)["data"]
    tree = build_tree(data_dir)
    return {"tree": tree}


@router.delete("/bots/{bot_id}/file/{file_path:path}")
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


@router.post("/bots/{bot_id}/upload")
async def upload(bot_id: str, request: Request, file: UploadFile, path: Optional[str] = None):
    require_bot_access(bot_id, password_from_request(request))
    return await store_upload(bot_id, file, path)


@router.post("/bots/{bot_id}/upload_zip")
async def upload_zip(bot_id: str, request: Request, file: UploadFile, path: Optional[str] = None):
    require_bot_access(bot_id, password_from_request(request))
    return await store_upload(bot_id, file, path)


@router.post("/bots/{bot_id}/folders")
async def create_folder(bot_id: str, request: Request):
    body = await request.json()
    require_bot_access(bot_id, password_from_request(request, body))
    path = body.get("path", "")
    target = resolve_data_path(bot_id, path)
    target.mkdir(parents=True, exist_ok=True)
    rel = target.relative_to(bot_paths(bot_id)["data"].resolve()).as_posix()
    return {"status": "created", "folder": rel}


@router.delete("/bots/{bot_id}/folders")
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


@router.post("/bots/{bot_id}/move_file")
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
