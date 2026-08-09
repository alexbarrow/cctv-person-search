import os

from dotenv import load_dotenv

DOTENV_PATH = ".env"

load_dotenv(DOTENV_PATH)

config = {
    "ip": os.getenv("IP"),
    "port": int(os.getenv("PORT")),
    "user": os.getenv("USER_CAM"),
    "password": os.getenv("PASSWORD_CAM"),
}
