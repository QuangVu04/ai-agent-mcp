import json
from pydantic import create_model
from langchain_core.tools import StructuredTool
from functools import partial

class ToolManager:
    def __init__(self, client):
        self.client = client
        self.type_map = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "array": list[str],
        }
        self._tools = []

    def register(self, tool):
        """Thêm tool thủ công (tool viết tay)."""
        self._tools.append(tool)

    async def load_from_mcp(self, tools_meta):
        """Tạo tool wrapper từ MCP server metadata."""
        for tool in tools_meta:
            name = tool["name"]
            description = tool["description"]
            schema = tool["input_schema"]["properties"]

            def make_caller(tool_name: str):
                async def _caller(**kwargs):
                    result = await self.client.call_tool(tool_name, kwargs)
                    if not result.content:
                        return None

                    normalized = []
                    for item in result.content:
                        text = getattr(item, "text", None)
                        if not text:
                            continue
                        try:
                            normalized.append(json.loads(text))
                        except json.JSONDecodeError:
                            normalized.append(text)

                    # Nếu chỉ có một phần tử thì trả trực tiếp
                    return normalized[0] if len(normalized) == 1 else normalized
                return _caller

            fields = {
                k: (self.type_map.get(v.get("type"), str), ...)
                for k, v in schema.items()
            }
            ArgsModel = create_model(f"{name}Args", **fields)

            wrapped_tool = StructuredTool.from_function(
                coroutine=make_caller(name),
                name=name,
                description=description,
                args_schema=ArgsModel,
            )
            self._tools.append(wrapped_tool)

    def list_tools(self):
        return self._tools
