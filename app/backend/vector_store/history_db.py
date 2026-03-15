"""
SQLite-based project history storage.
Replaces the JSON file with a proper relational database.

Database file: data/history.db (auto-created)
Table: projects (id, title, type, prompt, design_json, chat_history_json, created_at)
"""
import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional

# Database path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
DB_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DB_DIR, "history.db")


def _get_connection() -> sqlite3.Connection:
    """Returns a SQLite connection, creating the DB and table if needed."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Access columns by name
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT '',
            prompt TEXT NOT NULL DEFAULT '',
            design_json TEXT NOT NULL DEFAULT '{}',
            chat_history_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Converts a database row to the dict format the UI expects."""
    return {
        "id": row["id"],
        "title": row["title"],
        "type": row["type"],
        "prompt": row["prompt"],
        "design": json.loads(row["design_json"]),
        "chat_history": json.loads(row["chat_history_json"]),
        "created_at": row["created_at"],
    }


# ==========================================
# PUBLIC API — Drop-in replacements
# ==========================================

def load_history() -> List[Dict[str, Any]]:
    """Loads all projects from the database, ordered by creation date."""
    try:
        conn = _get_connection()
        rows = conn.execute("SELECT * FROM projects ORDER BY id ASC").fetchall()
        conn.close()
        return [_row_to_dict(r) for r in rows]
    except Exception as e:
        print(f"[HistoryDB] Load error: {e}")
        return []


def save_project(project: Dict[str, Any]) -> int:
    """
    Inserts a new project into the database.
    Returns the new row ID.
    """
    try:
        conn = _get_connection()
        cursor = conn.execute(
            """INSERT INTO projects (title, type, prompt, design_json, chat_history_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                project.get("title", "Untitled"),
                project.get("type", ""),
                project.get("prompt", ""),
                json.dumps(project.get("design", {}), ensure_ascii=False),
                json.dumps(project.get("chat_history", []), ensure_ascii=False),
                datetime.now().isoformat(),
            )
        )
        conn.commit()
        new_id = cursor.lastrowid
        conn.close()
        print(f"[HistoryDB] Saved project id={new_id}: {project.get('title', '')}")
        return new_id
    except Exception as e:
        print(f"[HistoryDB] Save error: {e}")
        return -1


def update_project(project_id: int, updates: Dict[str, Any]):
    """
    Updates specific fields of an existing project.
    Accepts any combination of: title, type, prompt, design, chat_history.
    """
    try:
        conn = _get_connection()
        set_clauses = []
        params = []

        if "title" in updates:
            set_clauses.append("title = ?")
            params.append(updates["title"])
        if "type" in updates:
            set_clauses.append("type = ?")
            params.append(updates["type"])
        if "prompt" in updates:
            set_clauses.append("prompt = ?")
            params.append(updates["prompt"])
        if "design" in updates:
            set_clauses.append("design_json = ?")
            params.append(json.dumps(updates["design"], ensure_ascii=False))
        if "chat_history" in updates:
            set_clauses.append("chat_history_json = ?")
            params.append(json.dumps(updates["chat_history"], ensure_ascii=False))

        if set_clauses:
            params.append(project_id)
            conn.execute(
                f"UPDATE projects SET {', '.join(set_clauses)} WHERE id = ?",
                params
            )
            conn.commit()
        conn.close()
    except Exception as e:
        print(f"[HistoryDB] Update error: {e}")


def delete_project(project_id: int):
    """Deletes a project by its database ID."""
    try:
        conn = _get_connection()
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        conn.commit()
        conn.close()
        print(f"[HistoryDB] Deleted project id={project_id}")
    except Exception as e:
        print(f"[HistoryDB] Delete error: {e}")


def get_project_count() -> int:
    """Returns the total number of projects."""
    try:
        conn = _get_connection()
        count = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def migrate_from_json(json_path: str) -> int:
    """
    One-time migration: imports projects from the old JSON file into SQLite.
    Returns the number of imported projects.
    """
    if not os.path.exists(json_path):
        return 0

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            old_history = json.load(f)
    except Exception:
        return 0

    if not old_history:
        return 0

    # Check if DB already has data (skip if already migrated)
    if get_project_count() > 0:
        print("[HistoryDB] Database already has data, skipping migration.")
        return 0

    imported = 0
    for project in old_history:
        result = save_project(project)
        if result > 0:
            imported += 1

    print(f"[HistoryDB] Migrated {imported} projects from JSON to SQLite.")
    return imported