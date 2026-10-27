# tests/test_phase1.py
"""Unit tests for Phase 1 – baseline LLM-only agent.

All tests use mock objects so no real API keys or network calls are needed.
Run with: python -m pytest tests/test_phase1.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is on the path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Provider selection tests
# ---------------------------------------------------------------------------

class TestProviderSelection:
    """BaselineAgent must pick the correct provider from LLM_PROVIDER env var."""

    def _make_agent(self, provider_name: str, env_vars: dict):
        """Helper: patch env and provider classes, return an agent instance."""
        from src.llm.base import LLMBase

        # Create a dummy provider that satisfies the abstract interface.
        DummyProvider = MagicMock(spec=LLMBase)
        DummyProvider.return_value.generate = MagicMock(return_value="ok")

        with patch.dict("os.environ", {"LLM_PROVIDER": provider_name, **env_vars}):
            # Patch provider classes where the agent imports them.
            with patch("src.agent.baseline.GeminiProvider", DummyProvider) as gm, \
                 patch("src.agent.baseline.GroqProvider", DummyProvider) as gq:
                from src.agent import baseline as mod
                import importlib
                importlib.reload(mod)
                agent = mod.BaselineAgent()
        return agent

    def test_defaults_to_gemini(self):
        """When LLM_PROVIDER=gemini is set, GeminiProvider is instantiated."""
        with patch.dict("os.environ", {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "fake"}):
            with patch("src.llm.provider_factory._instantiate") as mock_inst:
                from src.llm.base import LLMBase
                dummy = MagicMock(spec=LLMBase)
                dummy.PROVIDER_NAME = "gemini"
                dummy.generate = MagicMock(return_value="hi")
                mock_inst.return_value = dummy
                from src.agent import baseline as mod
                import importlib; importlib.reload(mod)
                mod.BaselineAgent()
                mock_inst.assert_called_once_with("gemini")

    def test_groq_selected(self):
        """When LLM_PROVIDER=groq, GroqProvider is instantiated."""
        with patch.dict("os.environ", {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "fake"}):
            with patch("src.llm.provider_factory._instantiate") as mock_inst:
                from src.llm.base import LLMBase
                dummy = MagicMock(spec=LLMBase)
                dummy.PROVIDER_NAME = "groq"
                dummy.generate = MagicMock(return_value="hi")
                mock_inst.return_value = dummy
                from src.agent import baseline as mod
                import importlib; importlib.reload(mod)
                mod.BaselineAgent()
                mock_inst.assert_called_once_with("groq")

    def test_unknown_provider_raises(self):
        with patch.dict("os.environ", {"LLM_PROVIDER": "unknown_xyz"}):
            with patch("src.llm.provider_factory._instantiate",
                       side_effect=ValueError("Unknown provider 'unknown_xyz'")):
                from src.agent import baseline as mod
                import importlib; importlib.reload(mod)
                with pytest.raises(ValueError):
                    mod.BaselineAgent()


# ---------------------------------------------------------------------------
# Prompt / history construction tests
# ---------------------------------------------------------------------------

class TestHistoryConstruction:
    """The agent must maintain conversation history correctly."""

    def _agent_with_mock_provider(self):
        mock_provider = MagicMock()
        mock_provider.generate = MagicMock(return_value="mocked reply")
        mock_provider.PROVIDER_NAME = "ollama"
        with patch.dict("os.environ", {"LLM_PROVIDER": "ollama"}):
            with patch("src.llm.provider_factory._instantiate", return_value=mock_provider):
                from src.agent import baseline as mod
                import importlib; importlib.reload(mod)
                agent = mod.BaselineAgent()
        agent.provider = mock_provider
        return agent, mock_provider

    def test_system_prompt_is_first_message(self):
        agent, _ = self._agent_with_mock_provider()
        assert agent.history[0]["role"] == "system"
        assert "ZentroMart" in agent.history[0]["content"]

    def test_user_message_appended(self):
        agent, _ = self._agent_with_mock_provider()
        agent.add_user_message("Hello")
        assert agent.history[-1] == {"role": "user", "content": "Hello"}

    def test_reply_appended_to_history(self):
        agent, mock_p = self._agent_with_mock_provider()
        agent.add_user_message("Hi")
        reply = agent.get_response()
        assert reply == "mocked reply"
        assert agent.history[-1] == {"role": "assistant", "content": "mocked reply"}

    def test_multi_turn_history_grows(self):
        agent, mock_p = self._agent_with_mock_provider()
        agent.add_user_message("Turn 1")
        agent.get_response()
        agent.add_user_message("Turn 2")
        agent.get_response()
        # system + user + assistant + user + assistant = 5 entries
        assert len(agent.history) == 5

    def test_reset_clears_history(self):
        agent, _ = self._agent_with_mock_provider()
        agent.add_user_message("Something")
        agent.get_response()
        agent.reset()
        assert len(agent.history) == 1
        assert agent.history[0]["role"] == "system"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Provider errors must now raise ProviderError (not silently return strings)."""

    def _agent_with_failing_provider(self, error: Exception):
        from src.llm.base import ProviderError
        # Wrap generic errors in ProviderError since that's what providers now raise.
        if not isinstance(error, ProviderError):
            pe = ProviderError("test", f"{type(error).__name__}: {error}")
        else:
            pe = error
        mock_provider = MagicMock()
        mock_provider.generate = MagicMock(side_effect=pe)
        mock_provider.PROVIDER_NAME = "test"
        with patch.dict("os.environ", {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "fake"}):
            with patch("src.llm.provider_factory._instantiate", return_value=mock_provider):
                from src.agent import baseline as mod
                import importlib; importlib.reload(mod)
                agent = mod.BaselineAgent()
        agent.provider = mock_provider
        return agent

    def test_timeout_raises_provider_error(self):
        """A timeout from the provider must raise ProviderError (not return a string)."""
        import httpx
        from src.llm.base import ProviderError
        agent = self._agent_with_failing_provider(httpx.TimeoutException("timed out"))
        agent.add_user_message("Hello")
        with pytest.raises(ProviderError):
            agent.get_response()

    # Backward-compat alias so tests read naturally.
    def test_timeout_returns_error_string(self):
        """Alias: verify ProviderError is raised on timeout (not silently swallowed)."""
        self.test_timeout_raises_provider_error()

    def test_generic_exception_raises_provider_error(self):
        """A generic RuntimeError from the provider must raise ProviderError."""
        from src.llm.base import ProviderError
        agent = self._agent_with_failing_provider(RuntimeError("boom"))
        agent.add_user_message("Hello")
        with pytest.raises(ProviderError):
            agent.get_response()

    # Backward-compat alias.
    def test_generic_exception_returns_error_string(self):
        self.test_generic_exception_raises_provider_error()


# ---------------------------------------------------------------------------
# Missing API key tests
# ---------------------------------------------------------------------------

class TestMissingApiKey:
    """Providers must raise ValueError with a clear message when keys are absent."""

    def test_gemini_raises_without_key(self):
        pytest.importorskip("google.genai", reason="google-genai not installed")
        import os
        saved = os.environ.pop("GEMINI_API_KEY", None)
        try:
            from src.llm import gemini as mod
            import importlib; importlib.reload(mod)
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                mod.GeminiProvider()
        finally:
            if saved is not None:
                os.environ["GEMINI_API_KEY"] = saved

    def test_groq_raises_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            import os; os.environ.pop("GROQ_API_KEY", None)
            from src.llm import groq as mod
            import importlib; importlib.reload(mod)
            with pytest.raises(ValueError, match="GROQ_API_KEY"):
                mod.GroqProvider()


# ---------------------------------------------------------------------------
# Groq provider HTTP mock tests
# ---------------------------------------------------------------------------

class TestGroqProvider:
    """Groq provider should parse the OpenAI-style JSON response correctly."""

    def test_successful_response_parsed(self):
        import json, os
        fake_response = {
            "choices": [{"message": {"content": "  Hello from Groq!  "}}]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status = MagicMock()

        with patch.dict("os.environ", {"GROQ_API_KEY": "fake"}):
            from src.llm import groq as mod
            import importlib; importlib.reload(mod)
            provider = mod.GroqProvider.__new__(mod.GroqProvider)
            provider._api_key = "fake"   # now private
            provider.model_name = "test-model"
            import httpx
            provider._client = MagicMock(spec=httpx.Client)  # now private
            provider._client.post.return_value = mock_resp

            result = provider.generate([{"role": "user", "content": "Hi"}])
        assert result == "Hello from Groq!"

    def test_http_error_raises_provider_error(self):
        """A network error must now raise ProviderError (not return an error string)."""
        import httpx
        from src.llm.base import ProviderError
        with patch.dict("os.environ", {"GROQ_API_KEY": "fake"}):
            from src.llm import groq as mod
            import importlib; importlib.reload(mod)
            provider = mod.GroqProvider.__new__(mod.GroqProvider)
            provider._api_key = "fake"
            provider.model_name = "test-model"
            provider._client = MagicMock(spec=httpx.Client)
            provider._client.post.side_effect = httpx.RequestError("connection refused")

            with pytest.raises(ProviderError):
                provider.generate([{"role": "user", "content": "Hi"}])

    # Backward-compat alias.
    def test_http_error_returns_error_string(self):
        self.test_http_error_raises_provider_error()


# ---------------------------------------------------------------------------
# Evaluation script sanity tests (no API calls)
# ---------------------------------------------------------------------------

class TestEvalScript:
    """Evaluation helpers must work without any API call."""

    def test_must_include_pass(self):
        from src.eval.baseline_eval import _check_must_include
        assert _check_must_include("Instant, within 2 hours", "instant / 2 hours") == "PASS"

    def test_must_include_fail(self):
        from src.eval.baseline_eval import _check_must_include
        assert _check_must_include("5 to 7 days", "instant / 2 hours") == "FAIL"

    def test_must_include_na(self):
        from src.eval.baseline_eval import _check_must_include
        assert _check_must_include("anything", "") == "N/A"

    def test_must_not_include_pass(self):
        from src.eval.baseline_eval import _check_must_not_include
        assert _check_must_not_include("No fee applies", "₹49") == "PASS"

    def test_must_not_include_fail(self):
        from src.eval.baseline_eval import _check_must_not_include
        assert _check_must_not_include("charge ₹49 fee", "₹49") == "FAIL"

    def test_load_questions_count(self):
        from src.eval.baseline_eval import load_questions
        qs = load_questions()
        assert len(qs) == 52

    def test_load_questions_limit(self):
        from src.eval.baseline_eval import load_questions
        qs = load_questions(limit=5)
        assert len(qs) == 5
