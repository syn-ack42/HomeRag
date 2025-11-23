"""Bot registry and access management routes."""
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request

from app_core.bot_registry import (
    create_bot,
    get_bot,
    is_bot_protected,
    load_bot_registry,
    password_from_request,
    remove_bot,
    require_bot_access,
    save_bot_registry,
    update_bot_password,
    verify_password,
)

router = APIRouter()


@router.get("/bots")
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


@router.post("/bots")
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


@router.post("/bots/{bot_id}/access")
async def access_bot(bot_id: str, request: Request):
    body = await request.json()
    require_bot_access(bot_id, password_from_request(request, body))
    bot = get_bot(bot_id)
    return {"status": "ok", "protected": is_bot_protected(bot)}


@router.post("/bots/{bot_id}/password")
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


@router.delete("/bots/{bot_id}")
async def delete_bot(bot_id: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    require_bot_access(bot_id, password_from_request(request, body))
    remove_bot(bot_id)
    return {"status": "deleted"}
