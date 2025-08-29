from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import httpx
import json
from bs4 import BeautifulSoup
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tools.send_email_tool import EmailTool

load_dotenv()

mcp = FastMCP("docs")
email_tool_instance = EmailTool()

USER_AGENT = "docs-app/1.0"
SERPER_URL="https://google.serper.dev/search"
SERPER_API_KEY=os.getenv("SERPER_API_KEY")

docs_urls = {
    "langchain": "python.langchain.com/docs",
    "llama-index": "docs.llamaindex.ai/en/stable",
    "openai": "platform.openai.com/docs",
}

async def search_web(query: str) -> dict | None:
    payload = json.dumps({"q": query, "num": 2})

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
            return response.json()
        except httpx.TimeoutException:
            return {"organic": []}
  
async def fetch_url(url: str):
  async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=30.0)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()
            return text
        except httpx.TimeoutException:
            return "Timeout error"

@mcp.tool()  
async def get_docs(query: str, library: str):
  """
  Search the latest docs for a given query and library.
  Supports langchain, openai, and llama-index.

  Args:
    query: The query to search for (e.g. "Chroma DB")
    library: The library to search in (e.g. "langchain")

  Returns:
    Text from the docs
  """
  if library not in docs_urls:
    raise ValueError(f"Library {library} not supported by this tool")
  
  query = f"site:{docs_urls[library]} {query}"
  results = await search_web(query)
  if len(results["organic"]) == 0:
    return "No results found"
  
  text = ""
  for result in results["organic"]:
    text += await fetch_url(result["link"])
  return text

@mcp.tool(description="Sends a plain text email using the configured SMTP server.")
async def send_email(from_address: str, to_addresses: list, subject: str, body: str):
  """
    Sends a plain text email.

    Connects to the configured SMTP server and sends an email with the provided
    details. The email is sent as 'text/plain'.

    Args:
        fromAddress (str): The email address of the sender.
                           Note: Some SMTP servers may require this to match the authenticated user.
        toAddresses (List[str]): A list of recipient email addresses.
        subject (str): The subject line of the email.
        body (str): The plain text content of the email body.

    Returns:
        Dict[str, str]: A dictionary indicating the status of the send operation.
            - "status" (str): "success" if the email was sent without raising an
              immediate error from the SMTP library.
    Raises:
        ValueError: If 'toAddresses' is not a non-empty list.
        ConnectionError: If connection to the SMTP server fails.
        Exception: For other errors during the email sending process (e.g., SMTP errors).
  """
  
  return email_tool_instance.send_text_email(from_address, to_addresses, subject, body)


if __name__ == "__main__":
    mcp.run(transport="stdio")