from pathlib import Path
from langchain_core.tools import tool
import json

INSTRUCTION_PATH = Path("instruction/users")

@tool
def update_user_preferences(preferences: list[str]) -> str:
    """Use this tool whenever the user explicitly sets or changes a persistent preference.

    Examples:
    - Response language (e.g., "always answer in Vietnamese")
    - Tone/style (e.g., "end each answer with 'my lord'")
    - Agent name (e.g., "your name is BigBoss")
    - Output format (e.g., "always answer in JSON")
    """
    user_file = INSTRUCTION_PATH / "user1.json"
    data = {"user": preferences}

    with open(user_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return f"User1 preferences updated successfully: {preferences}"
