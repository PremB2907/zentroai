# src/agent/baseline.py
"""Baseline LLM-only agent.
It maintains a simple multi-turn chat history and delegates generation to a
configured provider. Default: Ollama (local, no API key required).
Cloud providers (Groq, NVIDIA, Gemini) are optional overrides via LLM_PROVIDER.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from a .env file if present (ignored by git).
load_dotenv()

from ..llm.provider_factory import load_provider
from ..llm.base import ProviderError


class BaselineAgent:
    """A lightweight chat agent that talks to an LLM.

    The agent:
    * Prepends a system prompt that defines its role as a ZentroMart
      customer‑support assistant.
    * Keeps the full message history for multi‑turn context.
    * Chooses the provider based on the ``LLM_PROVIDER`` env variable.
      Default: ``ollama`` (local, no API key). Cloud providers require API keys.
    """

    SYSTEM_PROMPT = (
        "You are an AI assistant for ZentroMart, a fictional Indian online store. "
        "Answer customer queries politely and concisely. "
        "Do NOT claim you have access to order details, policies, or data that you "
        "haven't been given. If you are unsure, admit uncertainty and suggest contacting a human."
    )

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        self.provider = self._load_provider()

    def _load_provider(self):
        """Instantiate the provider based on ``LLM_PROVIDER``.
        Default provider is Ollama (local). Supported values: ``ollama``,
        ``groq``, ``gemini``, ``nvidia``.
        Raises ``ValueError`` with a clear message if configuration is missing.
        """
        return load_provider()   # reads LLM_PROVIDER, defaults to 'ollama'

    def add_user_message(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def get_response(self) -> str:
        """Send the accumulated history to the provider and return the reply.

        Raises :class:`ProviderError` on API failure so the evaluator and
        CLI can record a genuine error instead of a silent error string.
        """
        reply = self.provider.generate(self.history)
        # Append assistant reply to history for multi-turn context.
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        """Reset the conversation to the initial system prompt.
        Useful for unit tests.
        """
        self.history = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        # Provider stays the same – no need to re‑instantiate.
