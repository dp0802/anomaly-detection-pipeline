"""Configuration loader.

Reads ``config/config.yaml`` once and exposes a typed accessor so the rest
of the pipeline never has to know where settings live.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load YAML configuration from disk.

    Args:
        path: Optional override path. Defaults to ``config/config.yaml``.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError:  If the YAML cannot be parsed.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    logger.debug("Loaded config from %s", config_path)
    return cfg


def get_section(name: str, path: str | Path | None = None) -> Dict[str, Any]:
    """Return a top-level section of the config or raise ``KeyError``."""
    cfg = load_config(path)
    if name not in cfg:
        raise KeyError(f"Missing config section: {name}")
    return cfg[name]


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a sensible default format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
