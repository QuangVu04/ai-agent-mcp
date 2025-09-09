from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from mcpserver.mcp_client import MCPClient
import os
from pathlib import Path
from tools.toolsManager import ToolManager
from instruction.instructionManager import InstructionManager
from tools.updateUserFact import update_user_fact, query_user_fact, query_user_facts
   

load_dotenv()   

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
SERVER_PATH=os.getenv("SERVER_PATH")
INSTRUCTION_PATH = Path("instruction/users")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    system_prompt: SystemMessage
    
async def build_ai_agent():
    client = MCPClient()
    instruction_manager = InstructionManager()
    await client.connect_to_server(SERVER_PATH)

    tools_meta = await client.fetch_tools()
    tool_manager = ToolManager(client)
    
    tool_manager.register(update_user_fact)
    tool_manager.register(query_user_fact)

    await tool_manager.load_from_mcp(tools_meta)

    wrapped_tools = tool_manager.list_tools()

    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY
    ).bind_tools(wrapped_tools)

    initial_state = {
        "messages": [], 
        "system_prompt": await instruction_manager.load_system_instructions(tools_meta) 
    }
    print(initial_state)

    async def model_call(state: AgentState) -> AgentState:

        response = await model.ainvoke([state["system_prompt"]] + state["messages"] )
        state["messages"].append(response)

        return state

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

    return graph.compile(), client, initial_state