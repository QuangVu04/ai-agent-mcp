from pathlib import Path
from langchain_core.tools import tool
import json
from instruction.instructionManager import InstructionManager

INSTRUCTION_PATH = Path("instruction/users")

instruction_manager = InstructionManager()

@tool
async def update_user_preferences(preferences: list[str]) -> str:
    """
    Update persistent user preferences and system prompt.
    """

    # Save preferences to JSON
    user_file = INSTRUCTION_PATH / "user1.json"
    data = {"user": preferences}
    with open(user_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return f"User preferences updated successfully: {preferences}"
