"""Warm-start helpers for Ollama models."""

from __future__ import annotations

import asyncio
import logging
from typing import Iterable, Set

import requests

from app_core.rag_engine import get_ollama_url


logger = logging.getLogger("homerag")


REQUEST_TIMEOUT = (5, 60)


async def _ping_embedding_model(model: str):
    payload = {"model": model, "input": "warm start"}
    try:
        await asyncio.to_thread(
            requests.post,
            f"{get_ollama_url()}/api/embeddings",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        logger.info("Warmed embedding model %s", model)
    except Exception:
        logger.warning("Failed to warm embedding model %s", model, exc_info=True)


async def _ping_llm_model(model: str):
    payload = {"model": model, "prompt": "ping", "options": {"num_predict": 1}}
    try:
        await asyncio.to_thread(
            requests.post,
            f"{get_ollama_url()}/api/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        logger.info("Warmed LLM model %s", model)
    except Exception:
        logger.warning("Failed to warm LLM model %s", model, exc_info=True)


async def warm_start_models(embedding_models: Iterable[str], llm_models: Iterable[str]):
    """Warm the provided models without blocking startup."""

    embedding_set: Set[str] = {model for model in embedding_models if model}
    llm_set: Set[str] = {model for model in llm_models if model}

    tasks = []
    for model in embedding_set:
        tasks.append(asyncio.create_task(_ping_embedding_model(model)))
    for model in llm_set:
        tasks.append(asyncio.create_task(_ping_llm_model(model)))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

