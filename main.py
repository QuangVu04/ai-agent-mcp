import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from aiAssistant import build_ai_agent

async def main():
    app, client, state = await build_ai_agent() 

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Bye!")
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
            print("Ignoring anyio cancel scope bug on shutdown")
        else:
            raise
