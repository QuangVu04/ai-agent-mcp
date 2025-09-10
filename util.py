import json
import yaml
from pathlib import Path
from typing import Any, Dict
AgentState = Dict[str, Any] 
from langchain_core.messages import messages_to_dict, messages_from_dict, SystemMessage

STATE_FILE = "agent_state.json"

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

def save_state(state: AgentState):
    serializable_state = state.copy()

    # convert messages
    serializable_state["messages"] = messages_to_dict(state["messages"])
    
    if "system_prompt" in serializable_state:
        serializable_state.pop("system_prompt")

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable_state, f, ensure_ascii=False, indent=2)


def load_state(initial_state: AgentState) -> AgentState:
    path = Path(STATE_FILE)
    if path.exists() and path.stat().st_size > 0:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

            if "messages" in raw:
                raw["messages"] = messages_from_dict(raw["messages"])

            # merge vào initial_state để vẫn có system_prompt
            return {**raw, "system_prompt": initial_state["system_prompt"]}
    return initial_state
