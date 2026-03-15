import uuid
from typing import Dict

class SessionManager:
    """
    Keeps mapping between user sessions and LangGraph thread IDs.
    """

    def __init__(self):
        self.sessions: Dict[str, str] = {}

    def get_thread(self, session_id: str) -> str:

        if session_id not in self.sessions:
            self.sessions[session_id] = str(uuid.uuid4())

        return self.sessions[session_id]


SESSION_MANAGER = SessionManager()