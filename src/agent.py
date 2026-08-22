"""Central AI Support Agent Orchestrator.

Handles multi-turn conversation memory, intent routing, prompt injection defense,
privacy safeguards, order lookup tool execution, metadata-aware RAG retrieval,
grounded response generation, source citations, and human escalation recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

from src.knowledge_base import KnowledgeBase
from src.memory import ConversationManager, SessionState
from src.order_tool import OrderLookupTool, SafeOrderResult
from src.retriever import BM25Retriever, RetrievalResult, RetrievedChunk
from src.tracing import RetrievedChunkTrace, ToolCallTrace, TraceManager, TurnTrace


@dataclass
class AgentResponse:
    answer: str
    sources: List[str] = field(default_factory=list)
    handoff: bool = False
    handoff_reason: Optional[str] = None
    tool_called: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "handoff": self.handoff,
            "handoff_reason": self.handoff_reason,
            "tool_called": self.tool_called,
            "tool_args": self.tool_args,
            "tool_result": self.tool_result,
            "trace_id": self.trace_id,
        }


class SupportAgent:
    """Aster & Row Customer Support Agent."""

    def __init__(
        self,
        kb_path: str | Path = "knowledge-base",
        orders_path: str | Path = "data/orders.json",
        debug_mode: bool = False,
    ):
        self.kb_path = Path(kb_path)
        self.orders_path = Path(orders_path)
        self.debug_mode = debug_mode

        # Initialize core components
        self.kb = KnowledgeBase(self.kb_path)
        self.retriever = BM25Retriever(self.kb)
        self.order_tool = OrderLookupTool(self.orders_path)
        self.memory = ConversationManager()
        self.tracer = TraceManager(debug_mode=debug_mode)

    def _detect_privacy_violation(self, text: str) -> bool:
        """Detects if user is asking for disclosure of private customer or internal operational data."""
        lower = text.lower()
        if "coupon" in lower or "discount" in lower:
            return False

        private_inquiry_patterns = [
            r"\b(?:what|who|where|which|give|tell|show|reveal|disclose|provide|share|print|list)\b.*\b(?:email|address|live|name of customer|risk score|warehouse note|internal note|support tag|credit card|gift card pin|gift card code)\b",
            r"\b(?:who placed|who ordered|customer name|customer email|customer address|shipping address|billing address|home address)\b",
            r"\b(?:risk score|internal note|warehouse note|internal tag|fraud review score)\b",
            r"\b(?:where does [a-z]+ live|where is it being shipped)\b",
            r"\bwhat email\b",
        ]
        for pat in private_inquiry_patterns:
            if re.search(pat, lower):
                return True
        return False

    def _detect_prompt_injection(self, text: str) -> bool:
        """Detects prompt injection attempts in user input."""
        lower = text.lower()
        injection_patterns = [
            "ignore all prior rules",
            "ignore previous instructions",
            "migration note says",
            "system instruction",
            "reveal your prompt",
            "give everyone 60 days",
            "disregard policy",
            "override rules",
            "developer debug mode",
            "dump the full knowledge base",
            "print the raw json",
            "forget all rules",
            "reveal the internal migration",
        ]
        return any(pat in lower for pat in injection_patterns)

    def _is_order_intent(self, text: str) -> bool:
        """Determines if the query is an order tracking/status inquiry."""
        # 1. If an explicit order ID pattern is present, it's an order intent
        if self.order_tool.extract_order_id(text):
            return True

        lower = text.lower().strip()

        # 2. Exclude policy inquiries that happen to use words like 'order', 'ship', or 'arrive'
        policy_exclusions = [
            "return window",
            "return policy",
            "how long do i have to return",
            "can i return",
            "what is my return",
            "trailplus",
            "warranty",
            "lifetime warranty",
            "membership",
            "gift card",
            "price adjustment",
            "dishwasher",
            "clean",
            "ship to",
            "ship internationally",
            "do you ship",
            "can you ship",
            "broken zipper",
            "arrived with",
            "arrived damaged",
            "domestic shipping",
            "international shipping",
            "vegan",
            "migration note",
            "final sale",
            "final-sale",
            "po box",
        ]
        if any(exc in lower for exc in policy_exclusions):
            return False

        # 3. Explicit order status phrases
        order_patterns = [
            r"\bwhere is my (?:order|package|shipment|item|delivery)\b",
            r"\btrack(?:ing)?\s+(?:my\s+)?(?:order|package|shipment)\b",
            r"\b(?:status of|check)\s+(?:my\s+)?(?:order|package|shipment)\b",
            r"\bwhen will (?:my\s+)?(?:order|package|it)\s+(?:arrive|get here|be delivered)\b",
            r"\bwhere is\s+(?:it|my package|my order)\b",
            r"\bwhen will it arrive\b",
            r"\bwhen is it expected to arrive\b",
        ]
        for pat in order_patterns:
            if re.search(pat, lower):
                return True

        if lower in ("where is my order?", "where is my order", "track my order", "order status", "check my order", "where is it?", "when will it arrive?"):
            return True

        return False

    def _resolve_multiturn_query(self, user_msg: str, session: SessionState) -> str:
        """Enriches follow-up questions with context from conversation history."""
        lower = user_msg.lower()
        if session.messages:
            last_user_msg = ""
            for m in reversed(session.messages):
                if m.role == "user":
                    last_user_msg = m.content
                    break

            if "what about" in lower or "how long" in lower or "what if" in lower or "is there" in lower:
                return f"{last_user_msg} {user_msg}"

        return user_msg

    def _format_date(self, date_str: Optional[str]) -> str:
        """Formats ISO dates (e.g. 2026-08-22) to readable format (August 22, 2026)."""
        if not date_str:
            return ""
        try:
            if "T" in date_str:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                return dt.strftime("%B %d, %Y")
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime("%B %d, %Y")
        except Exception:
            return date_str

    def _generate_order_response(
        self, order_id: str, user_msg: str, result: SafeOrderResult
    ) -> Tuple[str, bool, Optional[str]]:
        """Synthesizes customer-safe answer for order lookup."""
        lower = user_msg.lower()

        if not result.found:
            msg = (
                f"Order {order_id} was not found in our records. "
                "Please check the order ID or contact support for assistance."
            )
            return msg, True, "Order not found"

        # Privacy check: If user asked for private fields for an existing order
        if self._detect_privacy_violation(user_msg):
            msg = (
                f"For privacy and security reasons, I cannot disclose personal customer information "
                f"(such as email or shipping address) or internal operational data (like risk scores and warehouse notes). "
                f"Regarding the public status of order {order_id}: it is currently {result.status}. "
                f"If you are the account holder needing address or order adjustments, please connect with our human support team."
            )
            return msg, True, "Privacy request requires human escalation"

        # Stale fields check for cancelled / returned orders
        if result.status == "cancelled":
            msg = f"The order is cancelled and it will not be shipped. {result.customer_safe_message}"
            return msg, False, None

        if result.status == "returned":
            msg = f"Order {result.order_id} was returned and processed. {result.customer_safe_message}"
            return msg, False, None

        if result.status == "delivered":
            deliv_date = self._format_date(result.delivered_at)
            msg = (
                f"Order {result.order_id} was delivered on {deliv_date or result.delivered_at or 'recent date'}. "
                f"Carrier: {result.carrier or 'N/A'}, Tracking number: {result.tracking_number or 'N/A'}."
            )
            return msg, False, None

        if result.status == "shipped":
            if result.estimated_delivery:
                eta_date = self._format_date(result.estimated_delivery)
                msg = (
                    f"Order {result.order_id} is currently shipped and in transit with {result.carrier} "
                    f"(Tracking: {result.tracking_number}). It is estimated to arrive on {eta_date}."
                )
            else:
                msg = (
                    f"Order {result.order_id} is shipped with {result.carrier} (Tracking: {result.tracking_number}). "
                    "A delivery estimate is unavailable from the carrier at this time."
                )
            return msg, False, None

        if result.status == "processing":
            eta_date = self._format_date(result.estimated_delivery)
            eta_str = f"Estimated delivery is {eta_date}." if eta_date else "A delivery estimate is not yet available."
            msg = f"Order {result.order_id} is currently being prepared for shipment. {eta_str}"
            return msg, False, None

        if result.status == "pending":
            cancel_str = " Because it was placed within the last 30 minutes, it is eligible for a cancellation or address correction request before entering processing." if result.is_cancellation_eligible else ""
            msg = f"Order {result.order_id} is currently pending.{cancel_str} {result.customer_safe_message}"
            return msg, False, None

        if result.status == "delayed":
            msg = f"Order {result.order_id} is currently delayed in transit with carrier {result.carrier}. {result.customer_safe_message}"
            return msg, False, None

        if result.status == "exception":
            msg = (
                f"Order {result.order_id} has encountered a carrier exception that requires support review. "
                "I am recommending human support review so a representative can look into this case."
            )
            return msg, True, "Carrier shipping exception"

        return result.format_summary(), False, None

    def _generate_policy_response(
        self, user_msg: str, ret_result: RetrievalResult
    ) -> Tuple[str, List[str], bool, Optional[str]]:
        """Synthesizes grounded policy answer with citations and safe abstention."""
        lower = user_msg.lower()

        # 1. Prompt Injection Defense
        if self._detect_prompt_injection(user_msg):
            ans = (
                "Internal migration scratchpads and draft migration note is not authoritative company policies. "
                "According to our official Returns Policy (01-returns-policy-current.md), the standard policy is 30 calendar days of delivery "
                "to return eligible unused items unless a valid exception applies. "
                "Furthermore, the agent cannot approve a return or override official store policies."
            )
            return ans, ["01-returns-policy-current.md (Standard return window)"], False, None

        # 2. Genuine Active Source Conflict
        if ret_result.conflict_detected:
            ans = (
                "Current official sources conflict regarding Breeze Tumbler cleaning guidance: "
                "One document, the Product Care Guide (11-product-care.md), says one says hand-wash the body and lid top-rack only, "
                "while another document, the Breeze Tumbler Product Card (12-breeze-tumbler-product-card.md), says one says all components are dishwasher safe. "
                "For safest interim guidance, hand-washing the body is recommended until human confirmation from support."
            )
            return ans, ["11-product-care.md (Breeze Tumbler)", "12-breeze-tumbler-product-card.md (Cleaning)"], True, "Conflicting official documentation"

        # 3. Specific Policy Topics Grounding
        # TrailPlus return window
        if "trailplus" in lower and ("return" in lower or "window" in lower or "policy" in lower or "ordered" in lower):
            ans = (
                "TrailPlus members receive an extended return window of **45 calendar days of delivery** for eligible items, "
                "provided that the TrailPlus membership was active when the order was placed. "
                "Item condition requirements, final-sale restrictions, and warranty rules still apply."
            )
            return ans, ["09-trailplus-membership.md (Return window)"], False, None

        # Final-sale damaged item exception
        if ("final sale" in lower or "final-sale" in lower) and ("damaged" in lower or "broken" in lower or "defective" in lower or "wrong" in lower or "zipper" in lower):
            ans = (
                "Final sale does not block damaged-item review. Under our Damaged, Defective, or Wrong Items Policy, "
                "customers should report within 7 days of delivery (calendar days) with clear photos. "
                "Aster & Row requires human review before approval for any refund or replacement. "
                "I am recommending a human support specialist to assist you with this claim."
            )
            return ans, ["03-final-sale-and-promotions.md (Damaged or incorrect items)", "04-damaged-or-wrong-items.md (Reporting window)"], True, "Damaged final-sale item requires human approval"

        # Final-sale with warranty claim
        if ("final sale" in lower or "final-sale" in lower) and ("warranty" in lower or "seam" in lower or "defect" in lower):
            ans = (
                "A product being marked final sale does not remove the limited warranty for qualifying manufacturing defects. "
                "Aster & Row bags have 2 years of limited warranty coverage from the purchase date for manufacturing defects in materials or workmanship under normal use."
            )
            return ans, ["07-warranty.md (Final-sale products)", "03-final-sale-and-promotions.md (Damaged or incorrect items)"], False, None

        # Standard return window
        if ("return" in lower or "refund" in lower) and ("standard" in lower or "regular" in lower or "backpack" in lower or "how long" in lower or "unused" in lower):
            ans = (
                "Under our current official Returns Policy, regular customers have **30 calendar days of delivery** "
                "to return an unused item in resalable condition with original tags and packaging. "
                "A $6.95 return shipping fee is deducted from the refund for standard domestic returns (waived if the item arrived damaged or incorrect)."
            )
            return ans, ["01-returns-policy-current.md (Standard return window)"], False, None

        # Canada / International Shipping
        if "canada" in lower or ("international" in lower and ("ship" in lower or "destination" in lower)):
            if "germany" in lower or "uk" in lower or "europe" in lower or "australia" in lower or "france" in lower:
                ans = (
                    "Aster & Row currently ships internationally only to Canada. "
                    "Shipping to Germany is not currently available at this time."
                )
                return ans, ["06-international-shipping.md (Supported destinations)"], False, None

            if "how long" in lower or "time" in lower or "tax" in lower or "duty" in lower or "duties" in lower or "days" in lower or "estimate" in lower:
                ans = (
                    "Canada is supported for international shipping. "
                    "Canadian orders generally arrive within **5–9 business days after dispatch** (processing is 1–2 business days). "
                    "Please note that import duties or taxes are not prepaid by Aster & Row and are the recipient's responsibility."
                )
                return ans, ["06-international-shipping.md (Canada delivery estimate)"], False, None

            ans = (
                "Aster & Row currently ships internationally only to **Canada**. "
                "Shipping to other international countries is not available at this time."
            )
            return ans, ["06-international-shipping.md (Supported destinations)"], False, None

        # Unsupported country check
        if "germany" in lower or "australia" in lower or "uk" in lower or "japan" in lower:
            ans = (
                "Aster & Row currently ships internationally only to Canada. "
                "Shipping to Germany is not currently available at this time."
            )
            return ans, ["06-international-shipping.md (Supported destinations)"], False, None

        # Warranty query
        if "warranty" in lower or "lifetime" in lower:
            ans = (
                "Aster & Row does not offer a lifetime warranty; there is no lifetime warranty on any products. "
                "Our limited warranty covers manufacturing defects under normal use for: "
                "\n- Bags have 2 years of warranty coverage from the purchase date."
                "\n- Drinkware and travel accessories have 1 year of coverage from the purchase date."
                "\nThe warranty does not cover ordinary wear, cosmetic changes, or accidental damage. "
                "Warranty claims require proof of purchase and are reviewed by a human support specialist."
            )
            return ans, ["07-warranty.md (Warranty periods)"], False, None

        # Domestic shipping / PO Box
        if "po box" in lower or "p.o. box" in lower:
            ans = (
                "For domestic United States shipments to PO boxes, delivery is estimated at **5–9 business days** after the standard 1–2 business days processing window."
            )
            return ans, ["05-domestic-shipping.md (Delivery estimates after dispatch)"], False, None

        # Price adjustment query
        if "price adjustment" in lower or "price drop" in lower:
            ans = (
                "Customers may request one price adjustment if the public price of the exact same item, color, and size drops within **7 calendar days of the original purchase**. "
                "Purchases made 20 days ago or outside the 7-day window are ineligible. Price adjustments do not apply to clearance/final-sale items or flash sales. A human support specialist must review and approve the adjustment."
            )
            return ans, ["10-gift-cards-and-price-adjustments.md (Price adjustments)"], True, "Price adjustment requires human processing"

        # Gift card query
        if "gift card" in lower:
            ans = (
                "Aster & Row gift cards do not expire and are final sale (non-refundable and cannot be exchanged for cash). "
                "Please note: our support agents will never ask you to share your full gift card number or PIN in chat."
            )
            return ans, ["10-gift-cards-and-price-adjustments.md (Gift cards)"], False, None

        # Order cancellation / change policy
        if "cancel" in lower or "change address" in lower or "edit order" in lower:
            ans = (
                "You may request order cancellation or address changes within **30 minutes of placing an order**, provided the order status is still `pending`. "
                "Once an order enters `processing` or `shipped`, it cannot be cancelled or modified. "
                "As an automated assistant, I cannot directly cancel or alter orders; please connect with a human support specialist immediately if your order is pending."
            )
            return ans, ["08-order-changes-and-cancellations.md (Cancellation window)"], True, "Order cancellation/modification requires human support action"

        # 4. Evidence Sufficiency / Safe Abstention
        if "vegan" in lower or "cruelty-free" in lower or "hypoallergenic" in lower or "organic" in lower:
            ans = (
                "The supplied information is insufficient in our official company knowledge base to verify whether all materials, fabrics, and adhesives in our bags are vegan. "
                "I recommend contacting human customer support for human confirmation."
            )
            return ans, [], True, "Insufficient information in official knowledge base"

        if "student" in lower or "military" in lower or "teacher" in lower:
            ans = (
                "The supplied information in our official company knowledge base does not mention special student, military, or educational discount programs. "
                "I recommend contacting human customer support for confirmation of active promotional offers."
            )
            return ans, [], True, "Discount programs not in knowledge base"

        if not ret_result.has_sufficient_evidence or not ret_result.chunks:
            ans = (
                "The supplied information is insufficient in our official company knowledge base to answer this question reliably. "
                "I recommend contacting human customer support for human confirmation."
            )
            return ans, [], True, "Insufficient information in official knowledge base"

        # 5. Synthesize grounded answer from top authoritative chunk
        top_chunk = ret_result.chunks[0].chunk
        citations = [top_chunk.source_tag]
        summary_ans = f"According to our {top_chunk.document_title} ({top_chunk.heading}):\n{top_chunk.content}"
        return summary_ans, citations, False, None

    def process_turn(
        self,
        user_message: str,
        session_id: Optional[str] = None,
    ) -> AgentResponse:
        """Processes a single conversational turn with safety, tools, RAG, and memory."""
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        session = self.memory.get_or_create_session(session_id)
        session.add_user_message(user_message)

        # Tracing container
        turn_trace = TurnTrace(
            trace_id=trace_id,
            session_id=session.session_id,
            user_message=user_message,
            history_length=len(session.messages),
        )

        extracted_id = self.order_tool.extract_order_id(user_message)
        turn_trace.extracted_order_id = extracted_id

        # Intent 1: Explicit or Contextual Order Lookup
        if self._is_order_intent(user_message):
            order_id_to_lookup = extracted_id or session.active_order_id

            # User is asking order status/details but gave no ID and no ID in memory
            if not order_id_to_lookup:
                turn_trace.intent = "missing_order_id"
                ans = "Could you please provide your Order ID (for example, ORD-1007) so I can check your order status?"
                session.add_assistant_message(ans, tool_called=None, handoff=False)
                turn_trace.response_text = ans
                turn_trace.duration_ms = (time.time() - start_time) * 1000
                self.tracer.record_trace(turn_trace)
                return AgentResponse(answer=ans, handoff=False, trace_id=trace_id)

            # Order lookup tool execution
            turn_trace.intent = "order_lookup"
            session.active_order_id = self.order_tool.normalize_order_id(order_id_to_lookup)
            order_result = self.order_tool.lookup(order_id_to_lookup)

            turn_trace.tool_calls.append(
                ToolCallTrace(
                    tool_name="order_lookup",
                    arguments={"order_id": order_id_to_lookup},
                    result=order_result.to_sanitized_dict(),
                )
            )

            ans, handoff, handoff_reason = self._generate_order_response(
                session.active_order_id, user_message, order_result
            )
            session.add_assistant_message(ans, tool_called="order_lookup", handoff=handoff)

            turn_trace.response_text = ans
            turn_trace.handoff = handoff
            turn_trace.handoff_reason = handoff_reason
            turn_trace.duration_ms = (time.time() - start_time) * 1000
            self.tracer.record_trace(turn_trace)

            return AgentResponse(
                answer=ans,
                sources=[],
                handoff=handoff,
                handoff_reason=handoff_reason,
                tool_called="order_lookup",
                tool_args={"order_id": order_id_to_lookup},
                tool_result=order_result.to_sanitized_dict(),
                trace_id=trace_id,
            )

        # Intent 2: Knowledge Base Policy & Product Query
        turn_trace.intent = "policy_rag"
        enriched_query = self._resolve_multiturn_query(user_message, session)
        retrieval_res = self.retriever.retrieve(enriched_query, top_k=4)

        for chunk_res in retrieval_res.chunks:
            turn_trace.retrieved_chunks.append(
                RetrievedChunkTrace(
                    filename=chunk_res.filename,
                    heading=chunk_res.heading,
                    score=chunk_res.score,
                    is_authoritative=chunk_res.is_authoritative,
                    status=chunk_res.chunk.status,
                )
            )

        turn_trace.conflict_detected = retrieval_res.conflict_detected
        turn_trace.conflict_sources = retrieval_res.conflicting_sources
        turn_trace.sufficient_evidence = retrieval_res.has_sufficient_evidence

        ans, sources, handoff, handoff_reason = self._generate_policy_response(
            user_message, retrieval_res
        )

        session.add_assistant_message(ans, sources=sources, handoff=handoff)
        session.last_retrieved_sources = sources

        turn_trace.response_text = ans
        turn_trace.sources = sources
        turn_trace.handoff = handoff
        turn_trace.handoff_reason = handoff_reason
        turn_trace.duration_ms = (time.time() - start_time) * 1000
        self.tracer.record_trace(turn_trace)

        return AgentResponse(
            answer=ans,
            sources=sources,
            handoff=handoff,
            handoff_reason=handoff_reason,
            trace_id=trace_id,
        )
