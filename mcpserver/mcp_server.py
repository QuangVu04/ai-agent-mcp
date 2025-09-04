from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import httpx
import json
from bs4 import BeautifulSoup
import os
import sys
from typing import Optional
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from tools.send_email_tool import EmailTool
from tools.google_calendar_tool import GoogleCalendarTool

load_dotenv()

mcp = FastMCP("docs")
email_tool_instance = EmailTool()
google_calendar_tool_instance = GoogleCalendarTool()

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

@mcp.tool()
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

@mcp.tool()
async def create_google_calendar_event(summary: str, location: str, description: str, start_datetime: str, end_datetime: str, attendees: list[str] = []):
    """
    Creates a Google Calendar event using the provided event details.

    Args:
        event (dict): A dictionary containing the event details as per Google Calendar API.
                      Example structure:
                      {
                          "summary": "Meeting with Bob",
                          "location": "123 Main St, Anytown, USA",
                          "description": "Discuss project updates.",
                          "start": {
                              "dateTime": "2023-10-01T10:00:00-07:00",
                              "timeZone": "America/Los_Angeles"
                          },
                          "end": {
                              "dateTime": "2023-10-01T11:00:00-07:00",
                              "timeZone": "America/Los_Angeles"
                          },
                          "attendees": [
                              {"email": "R0A5M@example.com"},
                              {"email": "o7PjI@example.com"}
                          ]
                      }

    Returns:
        str: A message indicating the result of the event creation.
    """

    return  google_calendar_tool_instance.insert_event(summary, location, description, start_datetime, end_datetime, attendees)

@mcp.tool()
async def list_upcoming_events(max_results: Optional[int] = 10,time_max: Optional[str] = None, time_min: Optional[str] = None):
    """
    Lists upcoming Google Calendar events.

    Args:
        time_min (Optional[str]): ISO start time. None = no limit.
        time_max (Optional[str]): ISO end time. None = no limit.
        max_results (Optional[int]): Maximum events. Default 10.

    Returns:
        List[dict]: A list of event details as per Google Calendar API.
    """
    max_results = int(max_results)
    return google_calendar_tool_instance.list_upcoming_events(max_results, time_max,time_min)

if __name__ == "__main__":
    mcp.run(transport="stdio")