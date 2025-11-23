"""RAG and chat-related routes."""
import asyncio
import json
import logging
import time

from chromadb.errors import InvalidArgumentError
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from app_core.bot_registry import DEFAULT_BOT_ID, password_from_request, require_bot_access
from app_core.rag_engine import (
    build_db,
    build_rag_chain,
    current_embedding_model,
    current_llm_model,
    ensure_db_embeddings,
    load_bot_config,
    load_prompt_template,
    load_system_prompt,
    log_performance,
    save_system_prompt,
)
from app_services.container import get_service_container

logger = logging.getLogger("homerag")

router = APIRouter()

service_container = get_service_container()
settings = service_container.settings


@router.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())


@router.post("/bots/{bot_id}/rebuild")
def rebuild(bot_id: str, request: Request):
    require_bot_access(bot_id, password_from_request(request))
    try:
        build_db(bot_id)
    except Exception as exc:
        logger.exception("Rebuild failed for bot %s", bot_id)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "reindexed"}


@router.get("/bots/{bot_id}/prompt")
def get_prompt(bot_id: str, request: Request):
    require_bot_access(bot_id, password_from_request(request))
    return {"prompt": load_system_prompt(bot_id)}


@router.post("/bots/{bot_id}/prompt")
async def update_prompt(bot_id: str, request: Request):
    body = await request.json()
    require_bot_access(bot_id, password_from_request(request, body))
    save_system_prompt(bot_id, body.get("prompt", ""))
    return {"status": "saved"}


@router.post("/ask_stream")
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
    ollama_url = str(settings.OLLAMA_URL)

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

    rag_context = build_rag_chain(
        bot_id=bot_id,
        question=question,
        config=bot_config,
        history_text=history_text,
        system_prompt=system_prompt,
    )
    embedding_model = rag_context["embedding_model"]
    rag_chain = rag_context["rag_chain"]

    async def event_stream():
        stream_start = time.perf_counter()
        tokens_emitted = 0
        try:
            async for token in rag_chain.astream(question):
                tokens_emitted += 1
                yield json.dumps({"type": "token", "token": token}) + "\n"
        except InvalidArgumentError:
            await asyncio.to_thread(build_db, bot_id, bot_config, [embedding_model])
            rebuilt_vectordb = rag_context["create_vectordb"]()
            rebuilt_chain = rag_context["build_chain"](rebuilt_vectordb)
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
