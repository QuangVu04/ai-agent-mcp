from pathlib import Path
from util import load_yaml
from util import load_json

INSTRUCTION_PATH = Path("instruction")

class InstructionManager:
    def __init__(self):
        self.instruction_path = INSTRUCTION_PATH
        
    def compile_instructions(self, user_id: str, domain_instructions: list[str]):
        system_level = load_yaml(self.instruction_path / "system.yaml").get("system", [])
        domain_level = load_yaml(self.instruction_path / "domain"/"domain.yaml").get("domain", [])
        user_level = load_json(self.instruction_path / "users" / f"{user_id}.json").get("user", [])

        final_prompt = "### System Instructions:\n"
        for instr in system_level:
            final_prompt += f"- {instr}\n"

        # final_prompt += "\n### Domain Instructions:\n"
        # all_domain_instr = domain_level + domain_instructions
        # for instr in all_domain_instr:
        #     final_prompt += f"- {instr}\n"

        final_prompt += "\n### User Instructions:\n"
        for instr in user_level:
            final_prompt += f"- {instr}\n"
            
        return final_prompt
    @staticmethod
    def build_domain_instructions(tools_meta: list[dict]) -> list[str]:
        domain_instructions = []
        for tool in tools_meta:
            name = tool["name"]
            desc = tool["description"]
            domain_instructions.append(f"Use tool `{name}` when: {desc}")
        return domain_instructions