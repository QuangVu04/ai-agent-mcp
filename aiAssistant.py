from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import asyncio
from mcpserver.mcp_client import MCPClient
import os
from pathlib import Path
from tools.toolsManager import ToolManager
from tools.updateUserPreferences import update_user_preferences
from instruction.instructionManager import InstructionManager    

load_dotenv()   

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
SERVER_PATH=os.getenv("SERVER_PATH")
INSTRUCTION_PATH = Path("instruction/users")

instruction_manager = InstructionManager()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
async def build_app():
    client = MCPClient()
    await client.connect_to_server(SERVER_PATH)

    tools_meta = await client.fetch_tools()
    tool_manager = ToolManager(client)

    tool_manager.register(update_user_preferences)

    await tool_manager.load_from_mcp(tools_meta)

    wrapped_tools = tool_manager.list_tools()

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
