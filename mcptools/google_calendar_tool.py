import logging
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import datetime
import os
from dotenv import load_dotenv

load_dotenv()

SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")
SCOPES = ["https://www.googleapis.com/auth/calendar"]
CALENDAR_ID = os.getenv("EMAIL_USER", "primary")


logging.basicConfig(level=logging.INFO)


class GoogleCalendarTool:
    def __init__(self):
        if not os.path.exists(SERVICE_ACCOUNT_FILE):
            raise FileNotFoundError(f"Service account file '{SERVICE_ACCOUNT_FILE}' not found.")
        self.creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )
        self.service = build("calendar", "v3", credentials=self.creds)

    def list_upcoming_events(self, max_results=None, time_max=None, time_min=None):
        try:
            now = datetime.datetime.now(datetime.UTC).isoformat()
            params = {"calendarId": CALENDAR_ID, "singleEvents": True, "orderBy": "startTime", "timeZone":"Asia/Ho_Chi_Minh",}
            if time_min:
                params["timeMin"] = time_min
            else:
                params["timeMin"] = now
            if time_max and time_min == None:
                params["timeMax"] = time_max
            if max_results:
                params["maxResults"] = max_results
            events_result = self.service.events().list(**params).execute()
            events = events_result.get("items", [])
            return events
        except HttpError as e:
            print("API call failed with HTTP error:", e)
            return e
        except Exception as e:
            print("Unexpected error:", e)
            return e
        
    def quick_add_event(self, text):
        try: 
            created_event = self.service.events().quickAdd(calendarId=CALENDAR_ID, text=text).execute()
            logging.info(f"Event created: {created_event.get('htmlLink')}")
        except HttpError as e:
            logging.error("API call failed with HTTP error:", e)
        except Exception as e:
            logging.error("Unexpected error:", e)
        
        
    def insert_event(self, summary: str, location: str, description: str, start_datetime: str, end_datetime: str, attendees: list[str] = []):
        try:
            event = {
                "summary": summary,
                "location": location,
                "description": description,
                "start": {
                    "dateTime": start_datetime,
                    "timeZone": "UTC",
                },
                "end": {
                    "dateTime": end_datetime,
                    "timeZone": "UTC",
                },
                "attendees": attendees,
            }
            created_event = self.service.events().insert(calendarId=CALENDAR_ID, body=event).execute()
            logging.info(f"Event created: {created_event.get('htmlLink')}")
            return created_event
        except HttpError as e:
            logging.error("API call failed with HTTP error:", e)
            return None
        except Exception as e:
            logging.error("Unexpected error:", e)
            return None
        