"""Shared service container for application dependencies."""

from .container import ServiceContainer, bootstrap_services, get_service_container

__all__ = ["ServiceContainer", "bootstrap_services", "get_service_container"]
