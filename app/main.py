import asyncio
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app_api.middleware import register_middleware
from app_api.routes import bots, files, models, rag
from app_config.settings import settings
from app_core.bot_registry import DEFAULT_PROMPT, bot_paths, load_bot_registry
from app_core.rag_engine import ensure_db_embeddings, load_bot_config, save_system_prompt
from app_core.warm_start import warm_start_models
from app_services.container import bootstrap_services

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("homerag")


bootstrap_services()


app = FastAPI(root_path=settings.ROOT_PATH)
app.mount("/static", StaticFiles(directory="static"), name="static")
register_middleware(app)

app.include_router(rag.router)
app.include_router(models.router)
app.include_router(bots.router)
app.include_router(files.router)


@app.on_event("startup")
async def startup_event():
    registry = load_bot_registry()
    app.state.startup_tasks = []
    embedding_models = set()
    llm_models = set()
    for bot in registry["bots"]:
        bot_id = bot["id"]
        paths = bot_paths(bot_id)
        bot_config = load_bot_config(bot_id)
        if not paths["prompt"].exists():
            save_system_prompt(bot_id, DEFAULT_PROMPT)
        embedding_models.add(bot_config.get("embedding_model"))
        if bot_config.get("last_built_embedding_model"):
            embedding_models.add(bot_config.get("last_built_embedding_model"))
        llm_models.add(bot_config.get("llm_model"))
        if any(paths["data"].iterdir()):
            task = asyncio.create_task(ensure_db_embeddings(bot_id))
            app.state.startup_tasks.append(task)
    warm_task = asyncio.create_task(warm_start_models(embedding_models, llm_models))
    app.state.startup_tasks.append(warm_task)
