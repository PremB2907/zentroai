# run.py
"""Entry point for the ZentroAI assistant.

Usage
-----
    # Full mode (default: RAG + tools + escalation)
    python run.py

    # RAG-only mode (no tools)
    python run.py --mode rag

    # Baseline LLM-only mode
    python run.py --mode baseline

    # Full mode with 8 evidence chunks
    python run.py --mode full --top-k 8
"""

from src.cli import main

if __name__ == "__main__":
    main()
