import asyncio
import logging
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Ensure project root is on the import path when running with --app-dir app
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_api.middleware import register_middleware
from app_api.routes import bots, files, models, rag
from app_config.settings import settings
from app_core.bot_registry import DEFAULT_PROMPT, bot_paths, load_bot_registry
from app_core.rag_engine import ensure_db_embeddings, load_bot_config, save_system_prompt
from app_services.container import bootstrap_services

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("homerag")


bootstrap_services()


app = FastAPI()
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
    for bot in registry["bots"]:
        bot_id = bot["id"]
        paths = bot_paths(bot_id)
        load_bot_config(bot_id)
        if not paths["prompt"].exists():
            save_system_prompt(bot_id, DEFAULT_PROMPT)
        if any(paths["data"].iterdir()):
            task = asyncio.create_task(ensure_db_embeddings(bot_id))
            app.state.startup_tasks.append(task)
