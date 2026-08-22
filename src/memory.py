"""Session-isolated multi-turn conversation memory.

Maintains turn history, active topic state, active order ID reference,
and provides context resolution across conversational turns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import uuid


@dataclass
class Message:
    role: str  # 'user', 'assistant', 'system'
    content: str
    sources: List[str] = field(default_factory=list)
    tool_called: Optional[str] = None
    handoff: bool = False


@dataclass
class SessionState:
    session_id: str
    messages: List[Message] = field(default_factory=list)
    active_order_id: Optional[str] = None
    active_topic: Optional[str] = None
    last_retrieved_sources: List[str] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(
        self,
        content: str,
        sources: List[str] = None,
        tool_called: Optional[str] = None,
        handoff: bool = False,
    ) -> None:
        self.messages.append(
            Message(
                role="assistant",
                content=content,
                sources=sources or [],
                tool_called=tool_called,
                handoff=handoff,
            )
        )

    def get_recent_history(self, max_turns: int = 5) -> List[Message]:
        return self.messages[-max_turns * 2 :] if self.messages else []

    def format_history_for_prompt(self, max_turns: int = 4) -> str:
        recent = self.get_recent_history(max_turns)
        if not recent:
            return "No previous conversation."
        lines = []
        for msg in recent:
            prefix = "Customer" if msg.role == "user" else "Support Agent"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)


class ConversationManager:
    """Manages active conversation sessions with memory isolation."""

    def __init__(self):
        self.sessions: Dict[str, SessionState] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> SessionState:
        if not session_id:
            session_id = str(uuid.uuid4())
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id=session_id)
        return self.sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            del self.sessions[session_id]
