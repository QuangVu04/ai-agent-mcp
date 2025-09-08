import json
from pydantic import create_model
from langchain_core.tools import StructuredTool

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
                    if result.content:
                        all_events = [json.loads(item.text) for item in result.content]
                        return all_events
                    return []
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
