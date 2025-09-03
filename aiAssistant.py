from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import asyncio
from mcpserver.mcp_client import MCPClient
from pydantic import create_model
from langchain_core.tools import StructuredTool
from bs4 import BeautifulSoup
import os

load_dotenv()   

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
SERVER_PATH=os.getenv("SERVER_PATH")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


async def build_app():
    client = MCPClient()
    await client.connect_to_server(SERVER_PATH)
    
    tools_meta = await client.fetch_tools()
    
    wrapped_tools = []
    type_map = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list[str],
    }
    for tool in tools_meta:
        name = tool["name"]
        description = tool["description"]
        schema = tool["input_schema"]["properties"]
        
        def clean_html(raw_html: str) -> str:
            return BeautifulSoup(raw_html, "html.parser").get_text(" ", strip=True)
        def make_caller(tool_name: str):
            async def _caller(**kwargs):
                print(f"Calling tool {tool_name} with args {kwargs}")
                result = await client.call_tool(tool_name, kwargs)
                if result.content:
                    raw_text = result.content[0].text
                    return clean_html(raw_text)
                return "No content"
            return _caller

        fields = {
            k: (type_map.get(v.get("type"), str), ...)
            for k, v in schema.items()
        }
        ArgsModel = create_model(f"{name}Args", **fields)

        wrapped_tool = StructuredTool.from_function(
            coroutine=make_caller(name), 
            name=name,
            description=description,
            args_schema=ArgsModel,
        )
        wrapped_tools.append(wrapped_tool)


    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY
    ).bind_tools(wrapped_tools)

    async def model_call(state: AgentState) -> AgentState:
        system_prompt = SystemMessage(content="You are my AI assistant.Khi nhận dữ liệu từ tool, hãy hướng dẫn từng bước.")
        response = await model.ainvoke([system_prompt] + state["messages"])
        return {"messages": [response]}

    def should_continue(state: AgentState): 
        last_message = state["messages"][-1]
        if not getattr(last_message, "tool_calls", None): 
            return "end"
        return "continue"

    graph = StateGraph(AgentState)
    graph.add_node("our_agent", model_call)
    graph.add_node("tools", ToolNode(tools=wrapped_tools))

    graph.set_entry_point("our_agent")

    graph.add_conditional_edges(
        "our_agent",
        should_continue,
        {"continue": "tools", "end": END},
    )

    graph.add_edge("tools", "our_agent")

    return graph.compile(), client



async def main():
    app, client = await build_app()

    try:
        inputs = {"messages": [("user", "search the latest langchain docs for chroma db")]}
        async for s in app.astream(inputs, stream_mode="values"):
            message = s["messages"][-1]
            if isinstance(message, (AIMessage, HumanMessage)):
                message.pretty_print()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cancel scope" in str(e):
            print("⚠️ Ignoring anyio cancel scope bug on shutdown")
        else:
            raise
