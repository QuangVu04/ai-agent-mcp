
import smtplib
from email.mime.text import MIMEText
from email.header import decode_header, make_header
import logging
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))



logging.basicConfig(level=logging.INFO)

class EmailTool:
    def __init__(self):
        self.user = EMAIL_USER
        self.password = EMAIL_PASS
        self.smtp_server = SMTP_SERVER
        self.smtp_port = SMTP_PORT

        if not self.user or not self.password:
            raise ValueError(
                "EMAIL_USER hoặc EMAIL_PASS chưa được set. "
                "Vui lòng điền vào .env hoặc biến môi trường."
            )
    

    def connect_smtp(self):
        server = smtplib.SMTP(self.smtp_server, self.smtp_port)
        server.starttls()
        server.login(self.user, self.password)
        return server

    def send_text_email(self, to_addresses: List[str], subject: str, body: str) -> Dict[str, str]:
        if not isinstance(to_addresses, list) or not to_addresses:
            raise ValueError("Input 'to_addresses' must be a non-empty list.")

      

        server = None
        try:
            msg = MIMEText(body, 'plain', 'utf-8')
            msg['Subject'] = make_header(decode_header(subject))
            msg['From'] = self.user
            msg['To'] = ', '.join(to_addresses)

            server = self.connect_smtp()
            server.sendmail(self.user, to_addresses, msg.as_string())
            logging.info(f"Text email sent successfully to {', '.join(to_addresses)}")
            return {"status": "success"}
        except Exception as e:
            logging.error(f"send_text_email failed: {e}")
            raise
        finally:
            if server:
                try:
                    server.quit()
                except Exception as e:
                    logging.warning(f"Error closing SMTP connection: {e}")
