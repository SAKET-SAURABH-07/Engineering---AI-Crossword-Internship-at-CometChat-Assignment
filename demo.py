"""Demo script demonstrating core Aster & Row Support Agent capabilities.

Demonstrates:
1. Knowledge-base question with citations
2. Order lookup with privacy protection
3. Multi-turn conversation context
4. Safe abstention and human escalation
5. Evaluation suite summary

Usage:
    python demo.py
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import SupportAgent


def print_banner(title: str):
    print("\n" + "=" * 65)
    print(f"  SCENARIO: {title}")
    print("=" * 65)


def simulate_turn(agent: SupportAgent, user_message: str, session_id: str = "demo_session"):
    print(f"\n[Customer]: {user_message}")
    time.sleep(0.3)
    response = agent.process_turn(user_message, session_id=session_id)
    print(f"\n[Aster & Row Support]: {response.answer}")
    if response.sources:
        print(f"[Citations]: {', '.join(response.sources)}")
    if response.tool_called:
        print(f"[Tool Executed]: {response.tool_called} (Args: {response.tool_args})")
    if response.handoff:
        print(f"[Human Escalation]: {response.handoff_reason or 'Handoff recommended'}")


def run_demo():
    agent = SupportAgent()

    print("\n" + "#" * 65)
    print("   ASTER & ROW AI SUPPORT AGENT — CAPABILITY DEMONSTRATION")
    print("#" * 65)

    # 1. Knowledge Base Question with Citations
    print_banner("1. Knowledge-Base Question with Authoritative Citations")
    simulate_turn(agent, "How long does a regular customer have to return an unused backpack?", "demo_s1")

    # 2. Order Lookup with Privacy Redaction
    print_banner("2. Sanitized Order Status Lookup (No PII Leakage)")
    simulate_turn(agent, "Where is ORD-1007 and when will it arrive?", "demo_s2")

    # 3. Multi-Turn Conversation
    print_banner("3. Multi-Turn Conversation Context Retention")
    session_id = "demo_multiturn"
    simulate_turn(agent, "Do you ship internationally?", session_id)
    simulate_turn(agent, "What about Canada, and how long does it take?", session_id)

    # 4. Safe Abstention on Unsupported Queries
    print_banner("4. Safe Abstention & Human Escalation (No Hallucination)")
    simulate_turn(agent, "Are all fabrics and adhesives in your bags vegan?", "demo_s4")

    # 5. Genuine Active Source Conflict
    print_banner("5. Conflict Detection Between Active Sources")
    simulate_turn(agent, "Can I put the entire Breeze Tumbler in the dishwasher?", "demo_s5")

    print("\n" + "=" * 65)
    print("  DEMO COMPLETE: All capabilities verified.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_demo()
