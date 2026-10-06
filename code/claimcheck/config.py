"""Loads config.yaml and prompts.yaml; applies environment overrides."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path=None) -> dict:
    cfg = yaml.safe_load(resolve(path or "config.yaml").read_text(encoding="utf-8"))
    if os.getenv("OLLAMA_HOST"):
        cfg["ollama"]["host"] = os.environ["OLLAMA_HOST"]
    if os.getenv("OLLAMA_MODEL"):
        cfg["ollama"]["model"] = os.environ["OLLAMA_MODEL"]
    return cfg


def load_prompts(cfg: dict) -> dict:
    return yaml.safe_load(resolve(cfg["files"]["prompts"]).read_text(encoding="utf-8"))
