from dotenv import load_dotenv
import os
import httpx
import json

load_dotenv()

SERPER_URL="https://google.serper.dev/search"
SERPER_API_KEY=os.getenv("SERPER_API_KEY")

async def search_web(query: str, num: int = 3) -> str:
    payload = json.dumps({"q": query, "num": num})
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                SERPER_URL, headers=headers, data=payload, timeout=30.0
            )
            response.raise_for_status()
            results = response.json()
        except httpx.TimeoutException:
            return "Timeout error."
    
    if not results or len(results.get("organic", [])) == 0:
        return "No results found."

    output = []
    for r in results["organic"]:
        output.append(f"- {r['title']} ({r['link']})\n{r.get('snippet', '')}\n")
    return "\n".join(output)

