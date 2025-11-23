"""Middleware registration for the FastAPI application."""
import logging
import time
from fastapi import FastAPI, Request

logger = logging.getLogger("homerag")


async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()

    body_bytes: bytes = b""
    try:
        body_bytes = await request.body()
        if body_bytes:
            max_len = 2000
            clipped = body_bytes[:max_len]
            suffix = " ...<truncated>" if len(body_bytes) > max_len else ""
            body_preview = clipped.decode("utf-8", errors="replace")
            logger.debug(
                "Received %s %s with body (%d bytes): %s%s",
                request.method,
                request.url.path,
                len(body_bytes),
                body_preview,
                suffix,
            )
        else:
            logger.debug("Received %s %s with empty body", request.method, request.url.path)
    except Exception as exc:  # pragma: no cover - defensive logging only
        logger.debug(
            "Received %s %s (failed to read body: %s)",
            request.method,
            request.url.path,
            exc,
        )

    async def receive() -> dict:
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    request_with_body = Request(request.scope, receive=receive)

    try:
        response = await call_next(request_with_body)
    except Exception as exc:
        logger.exception("Request %s %s failed: %s", request.method, request.url.path, exc)
        raise

    duration_ms = (time.perf_counter() - start_time) * 1000
    logger.debug(
        "Handled %s %s in %.2f ms with status %s",
        request.method,
        request.url.path,
        duration_ms,
        response.status_code,
    )
    response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"
    return response


def register_middleware(app: FastAPI) -> None:
    """Register application middleware on the provided app instance."""
    app.middleware("http")(log_requests)
