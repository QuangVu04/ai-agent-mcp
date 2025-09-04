from pathlib import Path
from util import load_yaml
from util import load_json

INSTRUCTION_PATH = Path("instruction")

class InstructionManager:
    def __init__(self):
        self.instruction_path = INSTRUCTION_PATH
        
    def compile_instructions(self, user_id: str, domain_instructions: list[str]):
        system_level = load_yaml(self.instruction_path / "system.yaml").get("system", [])
        user_level = load_json(self.instruction_path / "users" / f"{user_id}.json").get("user", [])
        print(f"Loaded user instructions for {user_id}: {user_level}")

        final_prompt = "### System Instructions:\n"
        for instr in system_level:
            final_prompt += f"- {instr}\n"

        # final_prompt += "\n### Domain Instructions:\n"
        # for instr in domain_instructions:
        #     final_prompt += f"- {instr}\n"

        final_prompt += "\n### User Instructions:\n"
        for instr in user_level:
            final_prompt += f"- {instr}\n"
        
        print(f"Compiled final prompt:\n{final_prompt}")
        return final_prompt