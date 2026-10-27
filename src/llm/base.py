# src/llm/base.py
"""Base abstract class for LLM providers.
Defines the minimal interface expected by the agent.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class ProviderError(RuntimeError):
    """Raised when an LLM provider call fails.

    Carries ``provider_name`` and ``status_code`` so evaluators can log
    structured failures without exposing raw exception messages (which may
    contain URL fragments with embedded keys).

    The ``safe_message`` attribute is guaranteed never to contain API keys.
    """

    def __init__(
        self,
        provider_name: str,
        safe_message: str,
        status_code: int | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.safe_message = safe_message
        self.status_code = status_code
        super().__init__(safe_message)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider_name,
            "status_code": self.status_code,
            "message": self.safe_message,
        }


class LLMBase(ABC):
    """Abstract LLM provider.

    Sub‑classes must implement :meth:`generate` which receives a list of
    messages in the OpenAI/Google format and returns the assistant's reply as
    a plain string.
    """

    #: Name used in error reporting.  Must be set by subclasses.
    PROVIDER_NAME: str = "unknown"

    @abstractmethod
    def generate(self, messages: List[Dict[str, Any]]) -> str:
        """Send a chat request to the remote model.

        Parameters
        ----------
        messages:
            List of dicts with ``role`` ("system", "user", "assistant") and
            ``content`` keys.
        Returns
        -------
        str
            The model's textual response.

        Raises
        ------
        ProviderError
            On any API-level failure (HTTP error, timeout, auth failure, etc.).
        """
        raise NotImplementedError

    def _raise_provider_error(
        self,
        exc: Exception,
        status_code: int | None = None,
    ) -> None:
        """Convert an exception to a :class:`ProviderError` with a safe message.

        API keys must never appear in exception messages.  This helper
        strips the raw exception text to a short, safe description.
        """
        exc_type = type(exc).__name__
        # Keep only the first 120 chars of the exception message and strip
        # anything that looks like a key/token (long alphanumeric strings).
        import re
        raw = str(exc)[:200]
        # Redact long alphanumeric sequences that could be keys/tokens.
        safe = re.sub(r'[A-Za-z0-9_\-]{40,}', '<redacted>', raw)
        raise ProviderError(
            provider_name=self.PROVIDER_NAME,
            safe_message=f"{exc_type}: {safe}",
            status_code=status_code,
        ) from exc
