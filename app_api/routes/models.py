"""Model listing and configuration routes."""
from fastapi import APIRouter, HTTPException, Request

from app_core.bot_registry import password_from_request, require_bot_access, save_bot_config
from app_core.rag_engine import (
    current_llm_model,
    fetch_installed_llm_models,
    fetch_model_summaries,
    load_bot_config,
)

router = APIRouter()


def _format_model_entry(entry: dict) -> dict:
    details = entry.get("details") or {}
    return {
        "name": entry.get("name"),
        "details": {
            "parameter_size": details.get("parameter_size") or details.get("size"),
            "quantization_level": details.get("quantization_level"),
            "context_length": details.get("context_length"),
            "embedding_length": details.get("embedding_length"),
        },
    }


@router.get("/models")
def list_models():
    summaries = fetch_model_summaries()
    models = [_format_model_entry(entry) for entry in summaries if not entry.get("is_embedding")]
    return {"models": models}


@router.get("/embedding-models")
def list_embedding_models():
    summaries = fetch_model_summaries()
    models = [_format_model_entry(entry) for entry in summaries if entry.get("is_embedding")]
    return {"models": models}


@router.get("/bots/{bot_id}/model")
def get_bot_model(bot_id: str, request: Request):
    require_bot_access(bot_id, password_from_request(request))
    return {"model": current_llm_model(bot_id)}


@router.post("/bots/{bot_id}/model")
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


@router.get("/bots/{bot_id}/config")
def get_bot_config(bot_id: str, request: Request):
    require_bot_access(bot_id, password_from_request(request))
    return {"config": load_bot_config(bot_id)}


@router.post("/bots/{bot_id}/config")
async def update_bot_config_endpoint(bot_id: str, request: Request):
    body = await request.json()
    require_bot_access(bot_id, password_from_request(request, body))
    current = load_bot_config(bot_id)
    current.update(body or {})
    save_bot_config(bot_id, current)
    return {"status": "saved", "config": load_bot_config(bot_id)}
