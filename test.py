import psycopg2
from config2 import load_config

try:
    config = load_config()
    conn = psycopg2.connect(**config)
    print("Connected to PostgreSQL successfully as", config["user"])
    conn.close()
except Exception as e:
    print("Connection failed:", e)
