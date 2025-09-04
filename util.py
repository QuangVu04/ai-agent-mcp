import json
import yaml
from pathlib import Path

def load_yaml(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_json(path: Path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}

def load_short_term_memory(SHORT_TERM_MEMORY_PATH) -> str:
    if SHORT_TERM_MEMORY_PATH.exists():
        return SHORT_TERM_MEMORY_PATH.read_text(encoding="utf-8")
    return ""

def save_short_term_memory(content: str, SHORT_TERM_MEMORY_PATH):
    SHORT_TERM_MEMORY_PATH.write_text(content, encoding="utf-8")