"""Adaptive image-level routing for custom object detection (disabled by default)."""

from .schemas import SCHEMA_VERSION
from .task_config import RoutingConfig, load_config

__all__ = ["SCHEMA_VERSION", "RoutingConfig", "load_config"]
