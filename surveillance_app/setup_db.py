import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared.db import init_db

init_db()
print("Database tables created successfully.")
