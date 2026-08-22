"""Deterministic, privacy-safe Order Lookup Tool.

Processes order lookups against data/orders.json with input normalization,
strict customer-data redaction, status precedence enforcement, and snapshot-based time logic.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional


@dataclass
class SafeOrderItem:
    name: str
    quantity: int
    final_sale: bool


@dataclass
class SafeOrderResult:
    found: bool
    order_id: Optional[str] = None
    status: Optional[str] = None
    membership_tier: Optional[str] = None
    items: List[SafeOrderItem] = field(default_factory=list)
    placed_at: Optional[str] = None
    status_updated_at: Optional[str] = None
    shipped_at: Optional[str] = None
    delivered_at: Optional[str] = None
    carrier: Optional[str] = None
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[str] = None
    customer_safe_message: Optional[str] = None
    is_cancellation_eligible: bool = False
    requires_handoff: bool = False
    error: Optional[str] = None

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """Returns a sanitized dict strictly containing only customer-safe fields."""
        if not self.found:
            return {
                "found": False,
                "order_id": self.order_id,
                "error": self.error or "Order not found.",
                "requires_handoff": self.requires_handoff,
            }

        return {
            "found": True,
            "order_id": self.order_id,
            "status": self.status,
            "membership_tier": self.membership_tier,
            "items": [
                {
                    "name": item.name,
                    "quantity": item.quantity,
                    "final_sale": item.final_sale,
                }
                for item in self.items
            ],
            "placed_at": self.placed_at,
            "status_updated_at": self.status_updated_at,
            "shipped_at": self.shipped_at,
            "delivered_at": self.delivered_at,
            "carrier": self.carrier,
            "tracking_number": self.tracking_number,
            "estimated_delivery": self.estimated_delivery,
            "customer_safe_message": self.customer_safe_message,
            "is_cancellation_eligible": self.is_cancellation_eligible,
            "requires_handoff": self.requires_handoff,
        }

    def format_summary(self) -> str:
        """Formatted human-readable summary for customer response."""
        if not self.found:
            return f"Order {self.order_id or 'unknown'} was not found in our records. Please double-check the order ID or contact customer support."

        item_names = ", ".join(f"{i.name} (Qty: {i.quantity})" for i in self.items)

        if self.status == "cancelled":
            return f"Order {self.order_id} ({item_names}) is cancelled and will not be shipped. {self.customer_safe_message}"

        if self.status == "returned":
            return f"Order {self.order_id} ({item_names}) was returned and processed. {self.customer_safe_message}"

        if self.status == "delivered":
            return f"Order {self.order_id} ({item_names}) was delivered on {self.delivered_at or 'recently'}. Carrier: {self.carrier or 'N/A'}, Tracking: {self.tracking_number or 'N/A'}."

        if self.status == "shipped":
            eta_str = f"Estimated delivery date is {self.estimated_delivery}." if self.estimated_delivery else "A delivery estimate is not currently available from the carrier."
            return f"Order {self.order_id} ({item_names}) has shipped with {self.carrier or 'the carrier'} (Tracking: {self.tracking_number or 'N/A'}). {eta_str} {self.customer_safe_message}"

        if self.status == "processing":
            eta_str = f"Estimated delivery is {self.estimated_delivery}." if self.estimated_delivery else "A delivery estimate is not yet available."
            return f"Order {self.order_id} ({item_names}) is currently processing. {eta_str} {self.customer_safe_message}"

        if self.status == "pending":
            cancel_str = " (Eligible for cancellation request within 30 minutes of order placement)" if self.is_cancellation_eligible else ""
            return f"Order {self.order_id} ({item_names}) is pending{cancel_str}. {self.customer_safe_message}"

        if self.status == "delayed":
            eta_str = f"Current estimated delivery is {self.estimated_delivery}." if self.estimated_delivery else ""
            return f"Order {self.order_id} ({item_names}) is delayed. Carrier: {self.carrier}. {eta_str} {self.customer_safe_message}"

        if self.status == "exception":
            return f"Order {self.order_id} has a shipping exception requiring support review. A human support representative will assist you."

        return f"Order {self.order_id} status: {self.status}. {self.customer_safe_message or ''}"


class OrderLookupTool:
    """Deterministic order lookup tool with sanitization and validation."""

    def __init__(self, orders_path: str | Path):
        self.orders_path = Path(orders_path)
        self.snapshot_at: datetime = datetime(2026, 8, 15, 12, 0, 0)
        self.orders_by_id: Dict[str, Dict[str, Any]] = {}
        self._load_orders()

    def _load_orders(self) -> None:
        if not self.orders_path.exists():
            return
        data = json.loads(self.orders_path.read_text(encoding="utf-8"))
        snapshot_str = data.get("snapshot_at", "2026-08-15T12:00:00Z")
        # Parse ISO datetime
        clean_snapshot = snapshot_str.replace("Z", "+00:00")
        try:
            self.snapshot_at = datetime.fromisoformat(clean_snapshot)
        except Exception:
            self.snapshot_at = datetime(2026, 8, 15, 12, 0, 0)

        for order in data.get("orders", []):
            order_id = order.get("order_id")
            if order_id:
                self.orders_by_id[order_id.strip().upper()] = order

    @staticmethod
    def extract_order_id(text: str) -> Optional[str]:
        """Extracts and normalizes order ID pattern from user text (e.g. 'ORD-1007')."""
        match = re.search(r"\b(ORD[-_]?[0-9]{3,5})\b", text, re.IGNORECASE)
        if match:
            raw = match.group(1).upper().replace("_", "-")
            if "-" not in raw and raw.startswith("ORD"):
                raw = f"ORD-{raw[3:]}"
            return raw
        return None

    @staticmethod
    def normalize_order_id(raw_id: str) -> str:
        """Normalizes harmless differences: whitespace, case, punctuation."""
        cleaned = raw_id.strip().upper()
        # Remove surrounding punctuation
        cleaned = re.sub(r"^[^A-Z0-9]+|[^A-Z0-9]+$", "", cleaned)
        match = re.search(r"ORD[-_]?([0-9]{3,5})", cleaned)
        if match:
            return f"ORD-{match.group(1)}"
        return cleaned

    def lookup(self, order_id_input: str) -> SafeOrderResult:
        """Performs safe, sanitized order lookup without exposing internal or private data."""
        if not order_id_input or not order_id_input.strip():
            return SafeOrderResult(
                found=False,
                error="Please provide a valid Order ID (e.g., ORD-1007).",
                requires_handoff=False,
            )

        norm_id = self.normalize_order_id(order_id_input)
        raw_order = self.orders_by_id.get(norm_id)

        if not raw_order:
            return SafeOrderResult(
                found=False,
                order_id=norm_id,
                error=f"Order {norm_id} was not found.",
                requires_handoff=True,
            )

        # Extract only customer-safe items
        safe_items = []
        for item in raw_order.get("items", []):
            safe_items.append(
                SafeOrderItem(
                    name=item.get("name", "Unknown Item"),
                    quantity=item.get("quantity", 1),
                    final_sale=bool(item.get("final_sale", False)),
                )
            )

        status = raw_order.get("status", "unknown").lower()
        placed_at = raw_order.get("placed_at")
        carrier = raw_order.get("carrier")
        tracking_number = raw_order.get("tracking_number")
        estimated_delivery = raw_order.get("estimated_delivery")
        customer_safe_msg = raw_order.get("customer_safe_message")
        membership_tier = raw_order.get("membership_tier", "standard")

        # Cancellation window calculation for pending status
        is_cancel_eligible = False
        if status == "pending" and placed_at:
            try:
                placed_dt = datetime.fromisoformat(placed_at.replace("Z", "+00:00"))
                # If placed within 30 minutes of snapshot_at
                elapsed_minutes = (self.snapshot_at - placed_dt).total_seconds() / 60.0
                if 0 <= elapsed_minutes <= 30:
                    is_cancel_eligible = True
            except Exception:
                is_cancel_eligible = False

        # Status precedence enforcement
        # Operational systems may retain stale delivery/carrier for cancelled/returned orders
        if status in ("cancelled", "returned"):
            carrier = None
            tracking_number = None
            estimated_delivery = None

        requires_handoff = False
        if status == "exception":
            requires_handoff = True

        return SafeOrderResult(
            found=True,
            order_id=norm_id,
            status=status,
            membership_tier=membership_tier,
            items=safe_items,
            placed_at=placed_at,
            status_updated_at=raw_order.get("status_updated_at"),
            shipped_at=raw_order.get("shipped_at"),
            delivered_at=raw_order.get("delivered_at"),
            carrier=carrier,
            tracking_number=tracking_number,
            estimated_delivery=estimated_delivery,
            customer_safe_message=customer_safe_msg,
            is_cancellation_eligible=is_cancel_eligible,
            requires_handoff=requires_handoff,
        )
