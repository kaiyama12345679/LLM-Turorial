import requests
from dotenv import load_dotenv
import os
import socket
load_dotenv()


def send_discord_webhook(message: str):
    url = os.getenv("DISCORD_WEBHOOK_URL")
    hostname = socket.gethostname()
    data = {
        "content": message,
        "username": f"{hostname}からの通知",
    }
    requests.post(url, json=data)