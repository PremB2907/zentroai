# src/llm/groq.py
"""Groq LLM provider using the Groq OpenAI-compatible REST API (via httpx).

Environment variables
---------------------
GROQ_API_KEY   Required.
GROQ_MODEL     Optional.  Defaults to ``llama-3.3-70b-versatile``.
               Other verified Groq models: ``llama-3.1-70b-versatile``,
               ``mixtral-8x7b-32768``, ``gemma2-9b-it``.
               Confirm availability for your account at console.groq.com.

Raises :class:`~src.llm.base.ProviderError` on any API-level failure so the
evaluator can count it as a genuine API error rather than a silent string.
"""

import os
from typing import List, Dict, Any

import httpx

from .base import LLMBase, ProviderError

_DEFAULT_MODEL = "llama-3.3-70b-versatile"
_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_TIMEOUT = 30.0   # seconds; Groq is usually <3 s


class GroqProvider(LLMBase):
    PROVIDER_NAME = "groq"

    def __init__(self, model_name: str | None = None) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file (see .env.example)."
            )
        self._api_key = api_key
        self.model_name = model_name or os.getenv("GROQ_MODEL", _DEFAULT_MODEL)
        self._client = httpx.Client(timeout=_TIMEOUT)

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.2,
        }
        # API key is in the header only — never in the body, never logged.
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            resp = self._client.post(_ENDPOINT, json=payload, headers=headers)
            status = resp.status_code
            if status != 200:
                # Do not surface the raw response body (may echo headers/keys).
                self._raise_provider_error(
                    Exception(f"HTTP {status} from Groq API"),
                    status_code=status,
                )
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except ProviderError:
            raise
        except Exception as exc:
            self._raise_provider_error(exc)
