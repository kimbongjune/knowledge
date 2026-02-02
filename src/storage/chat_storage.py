"""
채팅 세션 저장소
SQLite 기반 채팅 기록 영구 저장
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import uuid


@dataclass
class ChatMessage:
    """채팅 메시지"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    sources: Optional[List[Dict]] = None  # AI 응답 시 참고 문서


@dataclass
class ChatSession:
    """채팅 세션"""
    id: str
    title: str
    collection_name: Optional[str]
    model_name: Optional[str]
    created_at: str
    updated_at: str
    message_count: int = 0
    user_id: Optional[str] = None  # 로그인 사용자 ID


@dataclass
class User:
    """사용자"""
    id: str
    username: str
    created_at: str


class ChatStorage:
    """SQLite 기반 채팅 저장소"""
    
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "chat_history.db"
        
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 사용자 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        # 세션 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                collection_name TEXT,
                model_name TEXT,
                user_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # 메시지 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        ''')
        
        # 기존 sessions 테이블에 user_id 컬럼 추가 (마이그레이션 - 인덱스 생성 전에 먼저 실행)
        try:
            cursor.execute('ALTER TABLE sessions ADD COLUMN user_id TEXT')
        except sqlite3.OperationalError:
            pass  # 이미 컬럼 존재
        
        # 인덱스 (마이그레이션 후 생성)
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)')
        
        conn.commit()
        conn.close()
    
    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    # === 사용자 관리 ===
    
    def create_user(self, username: str, password_hash: str) -> Optional[User]:
        """사용자 생성"""
        import hashlib
        user_id = hashlib.md5(username.encode()).hexdigest()[:8]
        now = datetime.now().isoformat()
        
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO users (id, username, password_hash, created_at)
                VALUES (?, ?, ?, ?)
            ''', (user_id, username, password_hash, now))
            conn.commit()
            conn.close()
            return User(id=user_id, username=username, created_at=now)
        except sqlite3.IntegrityError:
            conn.close()
            return None  # 이미 존재하는 사용자
    
    def get_user_by_username(self, username: str) -> Optional[tuple]:
        """사용자 조회 (username으로)"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, password_hash, created_at FROM users WHERE username = ?', (username,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return (row['id'], row['username'], row['password_hash'], row['created_at'])
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """사용자 조회 (id로)"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, created_at FROM users WHERE id = ?', (user_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return User(id=row['id'], username=row['username'], created_at=row['created_at'])
        return None
    
    # === 세션 관리 ===
    
    def create_session(self, title: str = "새 대화", collection_name: str = None, model_name: str = None, user_id: str = None) -> ChatSession:
        """새 세션 생성"""
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sessions (id, title, collection_name, model_name, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, title, collection_name, model_name, user_id, now, now))
        conn.commit()
        conn.close()
        
        return ChatSession(
            id=session_id,
            title=title,
            collection_name=collection_name,
            model_name=model_name,
            created_at=now,
            updated_at=now,
            message_count=0,
            user_id=user_id
        )

    
    def list_sessions(self, limit: int = 50, user_id: str = None) -> List[ChatSession]:
        """세션 목록 (최근순, 사용자별 필터링)"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute('''
                SELECT s.*, COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                WHERE s.user_id = ?
                GROUP BY s.id
                ORDER BY s.updated_at DESC
                LIMIT ?
            ''', (user_id, limit))
        else:
            cursor.execute('''
                SELECT s.*, COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.id = m.session_id
                GROUP BY s.id
                ORDER BY s.updated_at DESC
                LIMIT ?
            ''', (limit,))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append(ChatSession(
                id=row['id'],
                title=row['title'],
                collection_name=row['collection_name'],
                model_name=row['model_name'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                message_count=row['message_count'],
                user_id=row['user_id'] if 'user_id' in row.keys() else None
            ))
        
        conn.close()
        return sessions
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """세션 조회"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.*, COUNT(m.id) as message_count
            FROM sessions s
            LEFT JOIN messages m ON s.id = m.session_id
            WHERE s.id = ?
            GROUP BY s.id
        ''', (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return ChatSession(
            id=row['id'],
            title=row['title'],
            collection_name=row['collection_name'],
            model_name=row['model_name'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            message_count=row['message_count'],
            user_id=row['user_id'] if 'user_id' in row.keys() else None
        )
    
    def update_session(self, session_id: str, title: str = None, collection_name: str = None, model_name: str = None) -> bool:
        """세션 정보 수정"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        updates = []
        values = []
        
        if title:
            updates.append("title = ?")
            values.append(title)
        if collection_name:
            updates.append("collection_name = ?")
            values.append(collection_name)
        if model_name:
            updates.append("model_name = ?")
            values.append(model_name)
        
        updates.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(session_id)
        
        cursor.execute(f'''
            UPDATE sessions SET {", ".join(updates)} WHERE id = ?
        ''', values)
        
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        
        return affected > 0
    
    def delete_session(self, session_id: str) -> bool:
        """세션 삭제 (메시지도 함께 삭제)"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
        
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        
        return affected > 0
    
    # === 메시지 관리 ===
    
    def add_message(self, session_id: str, role: str, content: str, sources: List[Dict] = None) -> bool:
        """메시지 추가"""
        now = datetime.now().isoformat()
        
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO messages (session_id, role, content, sources, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, role, content, json.dumps(sources) if sources else None, now))
        
        # 세션 업데이트 시간 갱신
        cursor.execute('UPDATE sessions SET updated_at = ? WHERE id = ?', (now, session_id))
        
        conn.commit()
        conn.close()
        
        return True
    
    def get_messages(self, session_id: str, limit: int = 100) -> List[ChatMessage]:
        """세션의 메시지 목록"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ?
        ''', (session_id, limit))
        
        messages = []
        for row in cursor.fetchall():
            messages.append(ChatMessage(
                role=row['role'],
                content=row['content'],
                timestamp=row['timestamp'],
                sources=json.loads(row['sources']) if row['sources'] else None
            ))
        
        conn.close()
        return messages
    
    def get_recent_messages(self, session_id: str, count: int = 10) -> List[ChatMessage]:
        """최근 N개 메시지 (컨텍스트 복원용)"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM (
                SELECT * FROM messages
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ) ORDER BY timestamp ASC
        ''', (session_id, count))
        
        messages = []
        for row in cursor.fetchall():
            messages.append(ChatMessage(
                role=row['role'],
                content=row['content'],
                timestamp=row['timestamp'],
                sources=json.loads(row['sources']) if row['sources'] else None
            ))
        
        conn.close()
        return messages
    
    def clear_messages(self, session_id: str) -> bool:
        """세션의 모든 메시지 삭제"""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
        
        conn.commit()
        conn.close()
        
        return True
    
    # === 자동 제목 생성 ===
    
    def auto_title_from_first_message(self, session_id: str) -> str:
        """첫 번째 메시지로 자동 제목 생성"""
        messages = self.get_messages(session_id, limit=1)
        if messages:
            title = messages[0].content[:30]
            if len(messages[0].content) > 30:
                title += "..."
            self.update_session(session_id, title=title)
            return title
        return "새 대화"
