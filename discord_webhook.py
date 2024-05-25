import requests
from dotenv import load_dotenv
import os
import psutil
import socket
load_dotenv()


def send_discord_webhook(message: str):
    url = str(os.getenv("DISCORD_WEBHOOK_URL"))
    hostname = socket.gethostname()
    pid = os.getpid()
    process = psutil.Process(pid)
    process_name = process.name()
    command = process.cmdline()
    data = {
        "content": f"process: {process_name}\ncommand: {command}\n\nmessage: {message}",
        "username": f"{hostname}からの通知",
    }
    requests.post(url, json=data)


if __name__ == "__main__":
    send_discord_webhook("Hello, Discord!")