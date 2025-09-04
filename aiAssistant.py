from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import asyncio
from mcpserver.mcp_client import MCPClient
from pydantic import create_model
from langchain_core.tools import StructuredTool
import os
import json
from pathlib import Path
from instructionManager import InstructionManager    
from util import load_short_term_memory, save_short_term_memory

load_dotenv()   

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
SERVER_PATH=os.getenv("SERVER_PATH")
INSTRUCTION_PATH = Path("instruction/users")
SHORT_TERM_MEMORY_PATH = Path("short_term_memory.txt")

instruction_manager = InstructionManager()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
@tool
def update_user_preferences(user_id: str, preferences: list[str]) -> str:
    """Cập nhật hoặc ghi đè các tùy chọn cá nhân của một user.

    Args:
        user_id: ID của người dùng (vd: 'user1', hiện tại mặc định đều là user1 )
        preferences: Danh sách sở thích mới (mỗi item là 1 rule hoặc preference)
    """
    user_file = INSTRUCTION_PATH / f"{user_id}.json"
    data = {"user": preferences}

    with open(user_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return f"User {user_id} preferences updated successfully: {preferences}"


async def build_app():
    client = MCPClient()
    await client.connect_to_server(SERVER_PATH)
    
    tools_meta = await client.fetch_tools()
    
    wrapped_tools = [update_user_preferences]
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
        
        def make_caller(tool_name: str):
            async def _caller(**kwargs):
                print(f"Calling tool {tool_name} with args {kwargs}")
                result = await client.call_tool(tool_name, kwargs)
                if result.content:
                    all_events = [json.loads(item.text) for item in result.content]
                    return all_events
                return []
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
        
        domain_instructions = instruction_manager.build_domain_instructions(tools_meta)

        final_prompt = instruction_manager.compile_instructions(
            user_id="user1",  # later replace with dynamic user id
            domain_instructions=domain_instructions
            )

        system_prompt = SystemMessage(
            content=f"{final_prompt}"
            )
        print("System prompt:", system_prompt)
        response = await model.ainvoke([system_prompt] + state["messages"])
        
        print("Model response:", response)
        previous_text = "\n".join([m.content for m in state["messages"]])
        new_content = previous_text + "\n" + response.content
        save_short_term_memory(new_content, SHORT_TERM_MEMORY_PATH)
        print(f"Updated short term memory: {state['messages']}")

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
        state = {"messages": []} 
        while True:
            user_input = input("👤 You: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Bye!")
                break

            state["messages"].append(HumanMessage(content=user_input))

            async for s in app.astream(state, stream_mode="values"):
                message = s["messages"][-1]
                state["messages"].append(message)
        
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
