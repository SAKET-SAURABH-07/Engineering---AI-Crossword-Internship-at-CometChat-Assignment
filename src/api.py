"""FastAPI Web Server for Aster & Row Support Agent.

Endpoints:
- POST /api/chat : Processes multi-turn customer chat messages
- GET /api/health : Returns service health status
- GET /api/orders/{order_id} : Returns sanitized order details
"""

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agent import SupportAgent

app = FastAPI(
    title="Aster & Row AI Support API",
    description="Customer support agent API with RAG, sanitized order lookups, and session management.",
    version="1.0.0",
)

agent = SupportAgent()


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    handoff: bool
    handoff_reason: Optional[str] = None
    tool_called: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Aster & Row AI Support Agent",
        "documents_loaded": len(agent.kb.documents),
        "chunks_indexed": len(agent.kb.chunks),
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    res = agent.process_turn(req.message, session_id=req.session_id)
    return ChatResponse(
        answer=res.answer,
        sources=res.sources,
        handoff=res.handoff,
        handoff_reason=res.handoff_reason,
        tool_called=res.tool_called,
        session_id=req.session_id,
        trace_id=res.trace_id,
    )


@app.get("/api/orders/{order_id}")
def lookup_order_endpoint(order_id: str):
    res = agent.order_tool.lookup(order_id)
    if not res.found:
        raise HTTPException(status_code=404, detail=f"Order {order_id} not found.")
    return res.to_sanitized_dict()
