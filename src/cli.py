# src/cli.py
"""Terminal REPL for ZentroAI.

Mode selection via ``--mode`` flag or ``AGENT_MODE`` env var:
  baseline  – LLM-only agent (Phase 1)
  rag       – retrieval-augmented agent, no tools (Phase 2)
  full      – RAG + order lookup + escalation (Phase 3, default)

Session commands:
  /exit   – quit
  /reset  – clear conversation history and state
  /mode   – show current agent mode
  /state  – (full mode only) show current conversation state
"""

import argparse
import os
import sys


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="zentroai",
        description="ZentroMart AI customer-support assistant.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "rag", "full"],
        default=os.getenv("AGENT_MODE", "full"),
        help=(
            "Agent mode: 'baseline' (LLM-only), 'rag' (retrieval-augmented), "
            "or 'full' (RAG + tools + escalation, default)."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of KB evidence chunks to retrieve per query.",
    )
    return parser.parse_args(argv)


def _make_agent(mode: str, top_k: int):
    if mode == "baseline":
        from .agent.baseline import BaselineAgent
        return BaselineAgent()
    elif mode == "rag":
        from .agent.rag_agent import RAGAgent
        return RAGAgent(top_k=top_k)
    else:  # full
        from .agent.dialogue_manager import DialogueManager
        return DialogueManager(top_k=top_k)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    mode = args.mode
    top_k = args.top_k

    print(f"ZentroAI – {mode.upper()} mode  (type /exit to quit, /reset to restart)")

    try:
        agent = _make_agent(mode, top_k)
    except (ValueError, RuntimeError) as exc:
        print(f"[Startup error] {exc}")
        sys.exit(1)

    print("Assistant: Hello! I'm ZentroMart's AI support assistant. How can I help you today?")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"/exit", "quit", "q"}:
            print("Goodbye!")
            break

        if user_input.lower() == "/reset":
            agent.reset()
            print("Conversation reset.")
            continue

        if user_input.lower() == "/mode":
            print(f"Current mode: {mode}")
            continue

        if user_input.lower() == "/state" and mode == "full":
            print(f"State: {agent.state.summary()}")
            continue

        # --- Dispatch ---
        if mode == "baseline":
            agent.add_user_message(user_input)
            reply = agent.get_response()
            print(f"Assistant: {reply}")

        elif mode == "rag":
            result = agent.chat(user_input)
            print(f"Assistant: {result.answer}")
            if result.sources:
                print(f"  [Sources: {', '.join(result.sources)}]")

        else:  # full
            result = agent.chat(user_input)
            print(f"Assistant: {result.answer}")
            if result.sources:
                print(f"  [Sources: {', '.join(result.sources)}]")
            if result.ticket_id:
                print(f"  [Ticket created: {result.ticket_id}]")


if __name__ == "__main__":
    main()
