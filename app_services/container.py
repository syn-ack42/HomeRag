"""Lightweight dependency injection container."""
from __future__ import annotations

from typing import Optional

from app_config.settings import Settings, settings as global_settings


class ServiceContainer:
    """Holds shared application-wide services and configuration."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings: Settings = settings or global_settings
        self._initialized = False

    def bootstrap(self):
        """Initialize lazy services. Extend as new shared dependencies are added."""
        if self._initialized:
            return
        # Placeholder for future shared service initialization
        self._initialized = True


container = ServiceContainer()


def bootstrap_services() -> ServiceContainer:
    """Ensure services are initialized and return the container."""
    container.bootstrap()
    return container


def get_service_container() -> ServiceContainer:
    """Access the global service container instance."""
    return container


__all__ = ["ServiceContainer", "bootstrap_services", "get_service_container", "container"]
