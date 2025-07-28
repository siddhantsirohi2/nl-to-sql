import os
from dotenv import load_dotenv

def load_config() -> dict:
    load_dotenv()
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD"),
        "dbname": os.getenv("DB_NAME", "postgres"),  # Default DB
    }
