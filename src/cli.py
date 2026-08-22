"""Interactive Command Line Interface for Aster & Row Support Agent.

Usage:
    python -m src.cli
    python -m src.cli --debug
"""

import argparse
import sys
from src.agent import SupportAgent


def main():
    parser = argparse.ArgumentParser(description="Aster & Row Customer Support Agent CLI")
    parser.add_argument("--debug", action="store_true", help="Enable structured debug tracing")
    args = parser.parse_args()

    agent = SupportAgent(debug_mode=args.debug)
    session_id = None

    print("\n=======================================================")
    print("   ASTER & ROW — AI CUSTOMER SUPPORT AGENT (CLI)     ")
    print("=======================================================")
    print("Type your message and press Enter.")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("Type 'clear' or 'new' to start a new conversation session.\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\nThank you for chatting with Aster & Row. Goodbye!")
                break

            if user_input.lower() in ("clear", "new", "reset"):
                session_id = None
                print("\n[Started new conversation session]")
                continue

            response = agent.process_turn(user_input, session_id=session_id)

            print(f"\nAster & Row Support: {response.answer}")

            if response.sources:
                print(f"\n[Sources: {', '.join(response.sources)}]")

            if response.handoff:
                print(f"[Human Handoff Recommended: {response.handoff_reason or 'Support assistance needed'}]")

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated.")
            break
        except Exception as e:
            print(f"\n[Error processing request: {e}]")


if __name__ == "__main__":
    main()
