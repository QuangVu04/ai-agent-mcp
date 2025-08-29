from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools = []

    async def connect_to_server(self, server_script_path: str) -> bool:
        """
        Connects to an MCP server via stdio.
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path])

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()
        return True

    async def fetch_tools(self):
        """
        List available tools from the MCP server.
        """
        if not self.session:
            raise RuntimeError("Session is not initialized")
        response = await self.session.list_tools()
        self.tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]
        return self.tools

    async def call_tool(self, tool_name: str, tool_args: dict):
        """
        Call a specific tool on the MCP server.
        """
        if not self.session:
            raise RuntimeError("Session is not initialized")
        return await self.session.call_tool(tool_name, tool_args)

    async def cleanup(self):
        """
        Close connections.
        """
        await self.exit_stack.aclose()
