"""Tests for deterministic Order Lookup Tool, normalization, privacy redaction, and status rules."""

import pytest
from src.order_tool import OrderLookupTool


@pytest.fixture
def order_tool():
    return OrderLookupTool("data/orders.json")


def test_order_id_normalization(order_tool):
    """Verifies that harmless input formatting differences are normalized."""
    res1 = order_tool.lookup("ord-1007")
    assert res1.found is True
    assert res1.order_id == "ORD-1007"

    res2 = order_tool.lookup(" ORD-1007. ")
    assert res2.found is True
    assert res2.order_id == "ORD-1007"

    res3 = order_tool.lookup("ord1007")
    assert res3.found is True
    assert res3.order_id == "ORD-1007"


def test_privacy_redaction(order_tool):
    """Verifies strict omission of customer personal details and internal operational notes."""
    res = order_tool.lookup("ORD-1007")
    sanitized = res.to_sanitized_dict()

    assert "customer" not in sanitized
    assert "internal" not in sanitized
    assert "risk_score" not in sanitized
    assert "warehouse_note" not in sanitized
    assert "support_tags" not in sanitized
    assert "email" not in sanitized
    assert "shipping_address" not in sanitized


def test_cancelled_order_stale_eta_suppressed(order_tool):
    """Verifies that cancelled orders do not expose stale carrier or ETA fields."""
    res = order_tool.lookup("ORD-1004")
    assert res.found is True
    assert res.status == "cancelled"
    assert res.estimated_delivery is None
    assert res.carrier is None
    assert res.tracking_number is None


def test_returned_order_stale_eta_suppressed(order_tool):
    """Verifies that returned orders do not expose stale ETA."""
    res = order_tool.lookup("ORD-1008")
    assert res.found is True
    assert res.status == "returned"
    assert res.estimated_delivery is None


def test_shipped_order_without_eta(order_tool):
    """Verifies that shipped order with missing ETA reports status without inventing date."""
    res = order_tool.lookup("ORD-1011")
    assert res.found is True
    assert res.status == "shipped"
    assert res.carrier == "Canada Post"
    assert res.estimated_delivery is None


def test_exception_order_flags_handoff(order_tool):
    """Verifies that orders with exception status trigger human handoff."""
    res = order_tool.lookup("ORD-1010")
    assert res.found is True
    assert res.status == "exception"
    assert res.requires_handoff is True


def test_pending_order_cancellation_window(order_tool):
    """Verifies 30-minute cancellation calculation against snapshot_at (2026-08-15T12:00:00Z)."""
    # ORD-1001 placed at 2026-08-15T11:45:00Z (15 min before snapshot) -> eligible
    res1 = order_tool.lookup("ORD-1001")
    assert res1.status == "pending"
    assert res1.is_cancellation_eligible is True

    # ORD-1002 status is processing -> not pending
    res2 = order_tool.lookup("ORD-1002")
    assert res2.status == "processing"
    assert res2.is_cancellation_eligible is False


def test_unknown_order_id(order_tool):
    """Verifies safe handling of unknown order IDs."""
    res = order_tool.lookup("ORD-9999")
    assert res.found is False
    assert res.requires_handoff is True
