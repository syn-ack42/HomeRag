"""Centralized application configuration."""
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATA_PATH: Path = Field(default=Path("/data"), env="DATA_PATH")
    DB_PATH: Path = Field(default=Path("/chroma"), env="DB_PATH")
    CONFIG_PATH: Path = Field(default=Path("/config"), env="CONFIG_PATH")
    BOT_CONFIG_ROOT: Optional[Path] = Field(default=None, env="BOT_CONFIG_ROOT")
    DEFAULT_EMBEDDING_MODEL: str = Field(
        default="mxbai-embed-large", env="EMBEDDING_MODEL"
    )
    DEFAULT_LLM_MODEL: str = Field(default="phi3", env="MODEL")
    OLLAMA_URL: str = Field(default="http://ollama:11434", env="OLLAMA_URL")
    EMBEDDING_MAX_WORKERS: int = Field(default=4, env="EMBEDDING_MAX_WORKERS")
    EMBEDDING_BATCH_SIZE: int = Field(default=8, env="EMBEDDING_BATCH_SIZE")
    EMBEDDING_CACHE_SIZE: int = Field(default=256, env="EMBEDDING_CACHE_SIZE")
    ROOT_PATH: str = Field(default="", env="ROOT_PATH")

    class Config:
        env_file = ".env"
        case_sensitive = False

    def model_post_init(self, __context):
        # Ensure BOT_CONFIG_ROOT follows CONFIG_PATH if not explicitly set
        if self.BOT_CONFIG_ROOT is None:
            object.__setattr__(self, "BOT_CONFIG_ROOT", Path(self.CONFIG_PATH) / "bots")
        root_path = (self.ROOT_PATH or "").strip()
        if root_path and not root_path.startswith("/"):
            root_path = "/" + root_path
        if root_path.endswith("/") and root_path != "/":
            root_path = root_path.rstrip("/")
        if root_path == "/":
            root_path = ""
        object.__setattr__(self, "ROOT_PATH", root_path)


settings = Settings()

__all__ = ["Settings", "settings"]
