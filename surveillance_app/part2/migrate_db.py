import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/surveillance",
)

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()
# Store embeddings as JSON text — pgvector server extension not required on Windows
cur.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS embedding TEXT;")
conn.commit()
cur.close()
conn.close()
print("Migration complete — embedding column added to events table.")
