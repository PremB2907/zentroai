# src/llm/gemini.py
"""Gemini provider using the current google-genai SDK (google.genai).

Environment variables
---------------------
GEMINI_API_KEY   Required.
GEMINI_MODEL     Optional.  Defaults to ``gemini-2.0-flash-001``
                 (stable-channel alias for the flash family, verified as
                 available via the Gemini Developer API in late 2026).
                 Override if your project uses a different model alias.
                 Do NOT use ``gemini-2.0-flash`` — that alias is deprecated
                 and returns HTTP 404.

Design notes
------------
* Uses ``google.genai`` (the new SDK that replaced ``google-generativeai``).
* We call ``client.models.generate_content`` with the full message list each
  turn so history lives entirely in the agent, not in a stateful SDK object.
* Temperature 0.2 for consistent evaluation results.
* Gemini's role for "assistant" messages is "model".
* Raises :class:`~src.llm.base.ProviderError` on any API-level failure.
"""

import os
from typing import List, Dict, Any

from google import genai
from google.genai import types

from .base import LLMBase, ProviderError

# Maps our internal role names to Gemini's expected role names.
_ROLE_MAP: dict[str, str] = {
    "user": "user",
    "assistant": "model",
    "system": "user",   # System instructions injected as a leading user turn.
}

_DEFAULT_MODEL = "gemini-2.0-flash-001"


class GeminiProvider(LLMBase):
    PROVIDER_NAME = "gemini"

    def __init__(self, model_name: str | None = None) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )
        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name or os.getenv("GEMINI_MODEL", _DEFAULT_MODEL)
        self._gen_config = types.GenerateContentConfig(temperature=0.2)

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        """Convert the message list to Gemini ``Content`` objects and call the
        API synchronously.

        System messages are prepended as a ``user`` turn because the SDK's
        ``system_instruction`` parameter requires the Messages API (v1beta).
        Injecting the system prompt as the first user turn is simpler and
        universally supported.

        Raises :class:`ProviderError` on any API failure, including the
        deprecated-model 404 error seen with ``gemini-2.0-flash``.
        """
        contents: list[types.Content] = []
        for msg in messages:
            role = _ROLE_MAP.get(msg["role"], "user")
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])],
                )
            )

        # Gemini rejects conversations that start with a "model" turn.
        if contents and contents[0].role == "model":
            contents.insert(
                0,
                types.Content(role="user", parts=[types.Part(text="(start)")]),
            )

        # Gemini also rejects consecutive same-role turns; merge them.
        merged: list[types.Content] = []
        for c in contents:
            if merged and merged[-1].role == c.role:
                prev_text = merged[-1].parts[0].text
                merged[-1] = types.Content(
                    role=c.role,
                    parts=[types.Part(text=prev_text + "\n" + c.parts[0].text)],
                )
            else:
                merged.append(c)

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=merged,
                config=self._gen_config,
            )
            return response.text.strip()
        except Exception as exc:
            # Extract HTTP status if available.
            status_code: int | None = None
            exc_str = str(exc)
            if "404" in exc_str:
                status_code = 404
            elif "429" in exc_str:
                status_code = 429
            elif "500" in exc_str:
                status_code = 500
            self._raise_provider_error(exc, status_code=status_code)
