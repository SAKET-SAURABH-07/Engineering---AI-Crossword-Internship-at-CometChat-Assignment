"""Structured observability and execution tracing for support agent turns.

Captures intent, retrieved passages, metadata, tool calls, sanitization,
final response, and handoff status in inspectable structured logs.
"""

from dataclasses import asdict, dataclass, field
import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agent.trace")


@dataclass
class RetrievedChunkTrace:
    filename: str
    heading: str
    score: float
    is_authoritative: bool
    status: str


@dataclass
class ToolCallTrace:
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]


@dataclass
class TurnTrace:
    trace_id: str
    session_id: str
    timestamp: float = field(default_factory=time.time)
    user_message: str = ""
    history_length: int = 0
    intent: str = ""
    extracted_order_id: Optional[str] = None
    retrieved_chunks: List[RetrievedChunkTrace] = field(default_factory=list)
    tool_calls: List[ToolCallTrace] = field(default_factory=list)
    conflict_detected: bool = False
    conflict_sources: List[str] = field(default_factory=list)
    sufficient_evidence: bool = True
    safety_triggers: List[str] = field(default_factory=list)
    response_text: str = ""
    sources: List[str] = field(default_factory=list)
    handoff: bool = False
    handoff_reason: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def pretty_log(self) -> None:
        """Emits formatted human-readable trace to logger."""
        summary = (
            f"[TRACE {self.trace_id[:8]}] Session: {self.session_id[:8]} | "
            f"Intent: {self.intent} | "
            f"Tools: {len(self.tool_calls)} | "
            f"Docs: {len(self.retrieved_chunks)} | "
            f"Handoff: {self.handoff} | "
            f"Time: {self.duration_ms:.1f}ms"
        )
        logger.info(summary)


class TraceManager:
    """Manages active turn traces and log dispatch."""

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.traces: List[TurnTrace] = []

    def record_trace(self, trace: TurnTrace) -> None:
        self.traces.append(trace)
        if self.debug_mode:
            trace.pretty_log()

    def get_last_trace(self) -> Optional[TurnTrace]:
        return self.traces[-1] if self.traces else None
