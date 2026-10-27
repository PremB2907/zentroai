# src/llm/nvidia.py
"""NVIDIA NIM LLM provider using the NVIDIA OpenAI-compatible inference API.

NVIDIA NIM exposes an OpenAI-compatible endpoint at
``https://integrate.api.nvidia.com/v1`` (NVIDIA AI Foundations, formerly
NVIDIA NGC).  Authentication is via an NVIDIA API key.

Environment variables
---------------------
NVIDIA_API_KEY   Required.
NVIDIA_MODEL     Optional.  Defaults to ``meta/llama-3.1-70b-instruct``.

**Model availability note**: NVIDIA NIM model identifiers depend on which
models are enabled for your API key / organisation plan.  The default
``meta/llama-3.1-70b-instruct`` is listed in the public NVIDIA API catalogue
as of mid-2026 but availability is not guaranteed for every account.  Override
``NVIDIA_MODEL`` in your ``.env`` file if a different model is provisioned.
Confirm availability at https://build.nvidia.com/explore/reasoning before the
live evaluation.

Raises :class:`~src.llm.base.ProviderError` on any API-level failure.
"""

import os
from typing import List, Dict, Any

import httpx

from .base import LLMBase, ProviderError

_DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"
_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
_TIMEOUT = 45.0   # NVIDIA NIM can be slower than Groq


class NvidiaProvider(LLMBase):
    PROVIDER_NAME = "nvidia"

    def __init__(self, model_name: str | None = None) -> None:
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. "
                "Add it to your .env file (see .env.example)."
            )
        self._api_key = api_key
        self.model_name = model_name or os.getenv("NVIDIA_MODEL", _DEFAULT_MODEL)
        self._client = httpx.Client(timeout=_TIMEOUT)

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = self._client.post(_ENDPOINT, json=payload, headers=headers)
            status = resp.status_code
            if status != 200:
                self._raise_provider_error(
                    Exception(f"HTTP {status} from NVIDIA NIM API"),
                    status_code=status,
                )
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except ProviderError:
            raise
        except Exception as exc:
            self._raise_provider_error(exc)
