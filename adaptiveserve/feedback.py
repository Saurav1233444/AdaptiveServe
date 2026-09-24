"""SQLite feedback makes monitoring persistent and concurrent writes atomic."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class FeedbackStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS feedback (id TEXT PRIMARY KEY, created TEXT, model TEXT, latency REAL, payload TEXT)")

    def connect(self):
        return sqlite3.connect(self.path, timeout=15)

    def record(self, result):
        payload = dict(result, timestamp=datetime.now(timezone.utc).isoformat())
        with self.connect() as db:
            db.execute("INSERT INTO feedback VALUES (?,?,?,?,?)", (payload["request_id"], payload["timestamp"],
                       payload["selected_model"], payload["latency_ms"], json.dumps(payload, allow_nan=False)))

    def metrics(self):
        with self.connect() as db:
            count, latency = db.execute("SELECT COUNT(*), AVG(latency) FROM feedback").fetchone()
            counts = dict(db.execute("SELECT model, COUNT(*) FROM feedback GROUP BY model"))
            recent = [json.loads(row[0]) for row in db.execute("SELECT payload FROM feedback ORDER BY created DESC LIMIT 60")]
        return {"requests": count, "average_latency_ms": latency, "current_model": recent[0]["selected_model"] if recent else None,
                "model_counts": counts, "recent": recent}
