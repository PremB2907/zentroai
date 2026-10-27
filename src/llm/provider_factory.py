# src/llm/provider_factory.py
"""Central provider factory with Groq → NVIDIA → Gemini fallback chain.

Provider priority (overridden by ``LLM_PROVIDER`` env var)
-----------------------------------------------------------
**Default: Ollama (local, no API key required)**

Cloud providers (Groq, NVIDIA, Gemini) remain available via explicit
``LLM_PROVIDER`` setting but are NOT called automatically when Ollama is
configured.  If Ollama is unavailable, ``ProviderError`` is raised
immediately — there is no silent cloud fallback from Ollama.

Usage
-----
Single provider (explicit, no fallback)::

    from src.llm.provider_factory import load_provider
    provider = load_provider()          # reads LLM_PROVIDER env var (default: ollama)
    provider = load_provider("groq")    # explicit cloud override

Provider chain with automatic fallback (cloud-only)::

    from src.llm.provider_factory import load_provider_chain
    provider, name = load_provider_chain()   # tries Groq → NVIDIA → Gemini
    answer = provider.generate(messages)     # raises ProviderError if all fail

Fallback rules
--------------
* Fallback in ``generate_with_fallback`` applies to cloud providers only.
* Ollama, if selected, is not eligible for cloud fallback.
* Fallback triggers only on :class:`ProviderError` (genuine API failure).
* Conversation history is passed unchanged to any fallback provider.
* API keys are never surfaced in error output.
"""


from __future__ import annotations

import os
import logging
from typing import List, Dict, Any

from .base import LLMBase, ProviderError

logger = logging.getLogger(__name__)

# Ordered fallback sequence used when LLM_PROVIDER is not set or is "auto".
_FALLBACK_ORDER = ["groq", "nvidia", "gemini"]


def _is_key_available(provider_name: str) -> bool:
    """Return True if the required API key / server env var is non-empty.

    For ``ollama``, always returns True (no key needed) — reachability is
    checked at generate-time by the ``OllamaProvider`` itself.
    """
    if provider_name == "ollama":
        return True   # no API key required; server checked at call time
    key_vars = {
        "groq": "GROQ_API_KEY",
        "nvidia": "NVIDIA_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    var = key_vars.get(provider_name, "")
    return bool(os.getenv(var, "").strip())


def _instantiate(provider_name: str) -> LLMBase:
    """Instantiate a provider by name.  Raises ValueError if key is missing."""
    if provider_name == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider()
    elif provider_name == "groq":
        from .groq import GroqProvider
        return GroqProvider()
    elif provider_name == "nvidia":
        from .nvidia import NvidiaProvider
        return NvidiaProvider()
    elif provider_name == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider()
    else:
        raise ValueError(
            f"Unknown provider {provider_name!r}. "
            "Choose: ollama, groq, nvidia, gemini."
        )


def load_provider(provider_name: str | None = None) -> LLMBase:
    """Load a single provider by name (no fallback).

    Parameters
    ----------
    provider_name:
        One of ``"ollama"``, ``"groq"``, ``"nvidia"``, ``"gemini"``.
        If ``None``, reads ``LLM_PROVIDER`` env var, defaulting to
        ``"ollama"`` (local inference, no API key required).

    Raises
    ------
    ValueError
        If the provider name is invalid or the required API key is missing.
    ProviderError
        (from OllamaProvider at generate-time) if the local server is down.
    """
    name = (provider_name or os.getenv("LLM_PROVIDER", "ollama")).lower()
    return _instantiate(name)


def load_provider_chain(
    priority: list[str] | None = None,
) -> tuple[LLMBase, str]:
    """Return the first available provider in the fallback chain.

    Tries providers in ``priority`` order, skipping any whose API key is not
    set.  Returns ``(provider_instance, provider_name)``.

    Parameters
    ----------
    priority:
        Ordered list of provider names.  Defaults to ``["groq", "nvidia",
        "gemini"]``.

    Raises
    ------
    RuntimeError
        If no provider in the chain has a configured API key.
    """
    chain = priority or _FALLBACK_ORDER
    errors: list[str] = []

    for name in chain:
        if not _is_key_available(name):
            errors.append(f"{name}: API key not set")
            continue
        try:
            provider = _instantiate(name)
            return provider, name
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}")

    raise RuntimeError(
        "No LLM provider could be initialised. "
        "Set at least one of GROQ_API_KEY, NVIDIA_API_KEY, GEMINI_API_KEY.\n"
        "Attempted: " + " | ".join(errors)
    )


def generate_with_fallback(
    messages: List[Dict[str, Any]],
    priority: list[str] | None = None,
    initial_provider: LLMBase | None = None,
    initial_provider_name: str = "groq",
) -> tuple[str, str]:
    """Call `generate` with automatic provider fallback.

    Tries providers in order.  Conversation history (``messages``) is
    passed unchanged to each provider — all three use the same OpenAI-style
    message format.

    Parameters
    ----------
    messages:
        Full message list including system prompt and history.
    priority:
        Override fallback order.  Defaults to ``["groq", "nvidia", "gemini"]``.
    initial_provider:
        If already instantiated, try this first before cycling the chain.
    initial_provider_name:
        Name label for ``initial_provider`` (used in logging/errors).

    Returns
    -------
    (answer, provider_name_used)

    Raises
    ------
    ProviderError
        If all providers fail.  The error carries the last failure's details.
    """
    chain = priority or _FALLBACK_ORDER
    last_error: ProviderError | None = None

    # Try the already-instantiated provider first.
    if initial_provider is not None:
        try:
            answer = initial_provider.generate(messages)
            return answer, initial_provider_name
        except ProviderError as exc:
            last_error = exc
            logger.warning(
                "Provider %s failed (%s). Trying fallback.",
                initial_provider_name,
                exc.safe_message,
            )

    # Walk the chain, skipping the initial provider and any without keys.
    for name in chain:
        if name == initial_provider_name:
            continue
        if not _is_key_available(name):
            continue
        try:
            provider = _instantiate(name)
            answer = provider.generate(messages)
            logger.info("Fallback provider %s succeeded.", name)
            return answer, name
        except (ProviderError, ValueError) as exc:
            last_error = exc if isinstance(exc, ProviderError) else ProviderError(
                provider_name=name,
                safe_message=f"ValueError: {exc}",
            )
            logger.warning(
                "Fallback provider %s failed (%s).",
                name,
                getattr(exc, "safe_message", str(exc)),
            )

    # All providers exhausted.
    if last_error is not None:
        raise last_error
    raise ProviderError(
        provider_name="all",
        safe_message="No providers available (API keys not configured).",
    )
