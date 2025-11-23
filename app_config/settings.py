"""Centralized application configuration."""
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


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

    class Config:
        env_file = ".env"
        case_sensitive = False

    def model_post_init(self, __context):
        # Ensure BOT_CONFIG_ROOT follows CONFIG_PATH if not explicitly set
        if self.BOT_CONFIG_ROOT is None:
            object.__setattr__(self, "BOT_CONFIG_ROOT", Path(self.CONFIG_PATH) / "bots")


settings = Settings()

__all__ = ["Settings", "settings"]
