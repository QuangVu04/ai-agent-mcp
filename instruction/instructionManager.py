from pathlib import Path
from util import load_yaml
from langchain_core.messages import SystemMessage
from tools.updateUserFact import query_user_facts


INSTRUCTION_PATH = Path("instruction")

class InstructionManager:
    def __init__(self):
        self.instruction_path = INSTRUCTION_PATH
        
    def compile_instructions(self, domain_instructions: list[str]):
        system_level = load_yaml(self.instruction_path / "system.yaml").get("system", [])
        domain_level = load_yaml(self.instruction_path / "domain"/"domain.yaml").get("domain", [])

        final_prompt = "### System Instructions:\n"
        for instr in system_level:
            final_prompt += f"- {instr}\n"

        final_prompt += "\n### Domain Instructions:\n"
        all_domain_instr = domain_level + domain_instructions
        for instr in all_domain_instr:
            final_prompt += f"- {instr}\n"
        
        final_prompt += "\n### User Instructions:\n"
        preference = query_user_facts (query="sở thích", category="preference")
        for instr in preference:
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
    
    async def load_system_instructions(self,tools_meta : list[dict]):
        domain_instructions = self.build_domain_instructions(tools_meta)

        final_prompt = self.compile_instructions(
            domain_instructions=domain_instructions
        )
        return SystemMessage(content=final_prompt)