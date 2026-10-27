# src/llm/ollama.py
"""Local Ollama LLM provider.

Uses the Ollama local HTTP API (``/api/chat``) — the chat endpoint that
accepts a full conversation history including system/user/assistant roles.
No API keys required.  Ollama must be running locally before the agent starts.

Environment variables
---------------------
OLLAMA_BASE_URL   Optional.  Defaults to ``http://localhost:11434``.
                  Override if Ollama is bound to a different port/host.
OLLAMA_MODEL      Optional.  Defaults to ``gemma2:2b`` (verified installed,
                  Oct 2026).  Set to any model tag returned by
                  ``ollama list`` / ``GET /api/tags``.

Installed models can be listed with::

    curl http://localhost:11434/api/tags

Error handling
--------------
* ``ProviderError`` is raised for:
  - Connection errors (server not running, wrong port)
  - Timeouts (connect or read)
  - HTTP errors (non-200 response)
  - Malformed / missing response fields
  - Model-not-found errors from Ollama
* Nothing is silently swallowed and returned as a fake assistant message.
* No secrets are present in Ollama's HTTP traffic (no auth header).

Conversation format
-------------------
Ollama's ``/api/chat`` endpoint accepts the same
``{"role": ..., "content": ...}`` message list as OpenAI's API.
The ``system``, ``user``, and ``assistant`` role names are all supported
natively — no translation needed.
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any

import httpx

from .base import LLMBase, ProviderError

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL    = "gemma2:2b"   # verified present in local Ollama install

# Timeouts: connect quickly (Ollama is local), but allow generous read time
# because smaller models on CPU can be slow to stream the first token.
_CONNECT_TIMEOUT = 5.0    # seconds — fast fail if server is not running
_READ_TIMEOUT    = 120.0  # seconds — allow up to 2 min for a local CPU response


class OllamaProvider(LLMBase):
    """LLM provider that calls a locally running Ollama server.

    Parameters
    ----------
    model_name:
        Ollama model tag (e.g. ``"gemma2:2b"``).  Overrides ``OLLAMA_MODEL``.
    base_url:
        Base URL of the Ollama server.  Overrides ``OLLAMA_BASE_URL``.
    """

    PROVIDER_NAME = "ollama"

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.base_url  = (base_url  or os.getenv("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)).rstrip("/")
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", _DEFAULT_MODEL)
        self._chat_url  = f"{self.base_url}/api/chat"
        self._client    = httpx.Client(
            timeout=httpx.Timeout(connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=30.0, pool=5.0)
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        """Send the full conversation history to Ollama and return the reply.

        Parameters
        ----------
        messages:
            List of ``{"role": ..., "content": ...}`` dicts.
            Supported roles: ``system``, ``user``, ``assistant``.

        Returns
        -------
        str
            The model's reply, stripped of leading/trailing whitespace.

        Raises
        ------
        ProviderError
            On any failure: connection error, timeout, HTTP error, or
            malformed response from Ollama.
        """
        payload = {
            "model":    self.model_name,
            "messages": messages,
            "stream":   False,   # get the full reply in one JSON response
            "options":  {"temperature": 0.2},
        }
        try:
            resp = self._client.post(self._chat_url, json=payload)
        except httpx.ConnectError as exc:
            raise ProviderError(
                provider_name=self.PROVIDER_NAME,
                safe_message=(
                    f"ConnectError: Ollama server not reachable at {self.base_url}. "
                    "Is 'ollama serve' running?"
                ),
                status_code=None,
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderError(
                provider_name=self.PROVIDER_NAME,
                safe_message=(
                    f"TimeoutException: Ollama did not respond within {_READ_TIMEOUT}s. "
                    f"Model: {self.model_name}"
                ),
                status_code=None,
            ) from exc
        except Exception as exc:
            self._raise_provider_error(exc)

        # Check HTTP status.
        if resp.status_code != 200:
            # Extract a short, safe error message from Ollama's JSON response.
            safe_detail = _extract_ollama_error(resp)
            raise ProviderError(
                provider_name=self.PROVIDER_NAME,
                safe_message=f"HTTP {resp.status_code}: {safe_detail}",
                status_code=resp.status_code,
            )

        # Parse the response body.
        try:
            data = resp.json()
        except Exception as exc:
            raise ProviderError(
                provider_name=self.PROVIDER_NAME,
                safe_message=f"MalformedResponse: could not parse Ollama JSON response",
                status_code=resp.status_code,
            ) from exc

        # Extract the assistant message content.
        try:
            content = data["message"]["content"]
            if not isinstance(content, str):
                raise KeyError("content not a string")
            return content.strip()
        except (KeyError, TypeError) as exc:
            raise ProviderError(
                provider_name=self.PROVIDER_NAME,
                safe_message=(
                    f"MalformedResponse: expected data['message']['content'], "
                    f"got keys={list(data.keys())}"
                ),
                status_code=resp.status_code,
            ) from exc

    # ------------------------------------------------------------------
    # Availability check (used by tests and the factory)
    # ------------------------------------------------------------------

    def is_server_reachable(self) -> bool:
        """Return True if the Ollama server is reachable (does not check model)."""
        try:
            r = self._client.get(f"{self.base_url}/api/tags")
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return the list of installed model tags, or [] on failure."""
        try:
            r = self._client.get(f"{self.base_url}/api/tags")
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_ollama_error(resp: httpx.Response) -> str:
    """Return a short, safe error string from an Ollama error response."""
    try:
        body = resp.json()
        # Ollama returns {"error": "..."} for model-not-found, etc.
        msg = body.get("error", "")
        # Truncate and strip anything that could be sensitive.
        return str(msg)[:120]
    except Exception:
        return f"(non-JSON body, status {resp.status_code})"
