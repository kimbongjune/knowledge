import sqlite3
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

class MetadataManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _is_valid_name(self, name: str) -> bool:
        """ChromaDB collection name validation"""
        # 3-512 chars, alphanumeric or ._-, start/end with alphanumeric
        if not name or len(name) < 3 or len(name) > 512:
            return False
        
        import re
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$', name):
            # Special case for 3-char names where regex might be tricky or simple check
            # Regex above requires at least 2 chars (start + end). 
            # If len is 3, middle can be anything valid.
            # Actually easier regex: ^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$
            return False
        return True

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS collections (
                        id TEXT PRIMARY KEY,
                        name TEXT UNIQUE NOT NULL,
                        created_at TEXT
                    )
                """)
        except Exception as e:
            logging.error(f"Failed to initialize metadata DB: {e}")
            raise

    def register_collection(self, name: str, physical_id: str = None) -> str:
        """
        Register a new collection.
        If physical_id is provided, uses it (migration).
        Otherwise generates a new UUID.
        Returns the physical ID.
        """
        if not physical_id:
            physical_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    "INSERT INTO collections (id, name, created_at) VALUES (?, ?, ?)",
                    (physical_id, name, datetime.now().isoformat())
                )
                return physical_id
            except sqlite3.IntegrityError:
                # Check idempotency
                cursor = conn.execute("SELECT id FROM collections WHERE name = ?", (name,))
                row = cursor.fetchone()
                if row and row[0] == physical_id:
                    return physical_id
                # Creating new collection with existing name?
                if row: 
                    raise ValueError(f"Collection name '{name}' already exists")
                # Using existing ID for different name?
                cursor = conn.execute("SELECT name FROM collections WHERE id = ?", (physical_id,))
                row = cursor.fetchone()
                if row:
                    raise ValueError(f"Physical ID '{physical_id}' already used by '{row[0]}'")
                raise

    def rename_collection(self, old_name: str, new_name: str):
        with sqlite3.connect(self.db_path) as conn:
            # Check target name availability
            cursor = conn.execute("SELECT 1 FROM collections WHERE name = ?", (new_name,))
            if cursor.fetchone():
                raise ValueError(f"Collection name '{new_name}' already exists")
            
            cursor = conn.execute("UPDATE collections SET name = ? WHERE name = ?", (new_name, old_name))
            if cursor.rowcount == 0:
                raise ValueError(f"Collection '{old_name}' not found")

    def delete_collection(self, name: str) -> Optional[str]:
        """Returns physical ID if found and deleted, None otherwise"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id FROM collections WHERE name = ?", (name,))
            row = cursor.fetchone()
            if not row:
                return None
            physical_id = row[0]
            
            conn.execute("DELETE FROM collections WHERE name = ?", (name,))
            return physical_id

    def get_physical_id(self, name: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id FROM collections WHERE name = ?", (name,))
            row = cursor.fetchone()
            return row[0] if row else None

    def list_collections(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT name FROM collections ORDER BY name")
            return [row[0] for row in cursor.fetchall()]

    def ensure_migration(self, physical_root: Path):
        """
        Scans physical directory for legacy folders not in DB.
        Registers them using folder name as both ID and Name.
        """
        # First, clean up any existing invalid entries in DB
        self.cleanup_invalid_entries()
        
        if not physical_root.exists():
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get all known physical IDs
                known_ids = set(row[0] for row in conn.execute("SELECT id FROM collections").fetchall())
                
                # Also check known names (legacy logic: name == id)
                known_names = set(row[0] for row in conn.execute("SELECT name FROM collections").fetchall())
                
                for item in physical_root.iterdir():
                    if item.is_dir():
                        folder_name = item.name
                        
                        # Validate name before migrating
                        if not self._is_valid_name(folder_name):
                            continue
                            
                        # Only migrate if neither name nor ID is known
                        if folder_name not in known_ids and folder_name not in known_names:
                            try:
                                conn.execute(
                                    "INSERT INTO collections (id, name, created_at) VALUES (?, ?, ?)",
                                    (folder_name, folder_name, datetime.now().isoformat())
                                )
                                print(f"Migrated legacy collection: {folder_name}")
                            except sqlite3.IntegrityError:
                                pass
        except Exception as e:
            logging.error(f"Migration failed: {e}")

    def cleanup_invalid_entries(self):
        """Remove entries from DB that violate naming rules"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT name FROM collections")
                to_delete = []
                for (name,) in cursor.fetchall():
                    if not self._is_valid_name(name):
                        to_delete.append(name)
                
                if to_delete:
                    print(f"Removing invalid collections from DB: {to_delete}")
                    for name in to_delete:
                        conn.execute("DELETE FROM collections WHERE name = ?", (name,))
        except Exception as e:
            logging.error(f"Cleanup failed: {e}")
