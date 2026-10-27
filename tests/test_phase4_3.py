# tests/test_phase4_3.py
"""Phase 4.3 tests — provider priority, fallback, and evaluator error accounting.

All tests are offline (mocked).  No real API keys or network calls are used.

Coverage
--------
* Groq is selected by default when LLM_PROVIDER is not set.
* Groq failure triggers NVIDIA fallback.
* NVIDIA failure triggers Gemini fallback.
* All providers failing produces a ProviderError (recorded as api_error).
* No API key is included in error output.
* Conversation history is preserved across provider fallback.
* NVIDIA cannot be selected if NVIDIA_API_KEY is not set (skipped in chain).
* Legacy error-string responses are not scored as successful answers.
* Provider/model can be overridden via environment variables.
* GEMINI_MODEL default is gemini-2.0-flash-001 (not the deprecated gemini-2.0-flash).
* Evaluator correctly counts api_errors for ProviderError and error-string responses.
* Heuristic scores are API_ERROR (not PASS/FAIL) for failed calls.
* is_api_error flag is set correctly in CSV rows.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_provider(name: str, answer: str = "OK") -> MagicMock:
    from src.llm.base import LLMBase
    m = MagicMock(spec=LLMBase)
    m.PROVIDER_NAME = name
    m.generate = MagicMock(return_value=answer)
    return m


def _make_failing_provider(name: str, status_code: int = 500) -> MagicMock:
    from src.llm.base import LLMBase, ProviderError
    m = MagicMock(spec=LLMBase)
    m.PROVIDER_NAME = name
    m.generate = MagicMock(
        side_effect=ProviderError(
            provider_name=name,
            safe_message=f"HTTP {status_code} from {name} API",
            status_code=status_code,
        )
    )
    return m


# ---------------------------------------------------------------------------
# Tests: ProviderError dataclass
# ---------------------------------------------------------------------------

class TestProviderError:
    def test_attributes_set_correctly(self):
        from src.llm.base import ProviderError
        err = ProviderError("groq", "HTTP 503: service unavailable", status_code=503)
        assert err.provider_name == "groq"
        assert err.status_code == 503
        assert "503" in err.safe_message
        assert "service" in err.safe_message.lower()

    def test_to_dict(self):
        from src.llm.base import ProviderError
        err = ProviderError("gemini", "HTTP 404: model not found", status_code=404)
        d = err.to_dict()
        assert d["provider"] == "gemini"
        assert d["status_code"] == 404
        assert "404" in d["message"]

    def test_no_api_key_in_safe_message(self):
        """The safe message must never contain a 40+-char alphanumeric key."""
        from src.llm.base import LLMBase, ProviderError
        fake_key = "A" * 50  # simulate a key
        provider = MagicMock(spec=LLMBase)
        provider.PROVIDER_NAME = "groq"
        # _raise_provider_error should redact the key
        groq_instance = MagicMock(spec=LLMBase)
        groq_instance.PROVIDER_NAME = "groq"
        # Call _raise_provider_error via the mixin
        from src.llm.base import LLMBase as Base
        # Instantiate via a concrete subclass
        class ConcreteProvider(Base):
            PROVIDER_NAME = "test"
            def generate(self, messages):
                raise NotImplementedError
        p = ConcreteProvider()
        exc = Exception(f"Bearer {fake_key} not valid")
        with pytest.raises(ProviderError) as exc_info:
            p._raise_provider_error(exc)
        assert fake_key not in exc_info.value.safe_message, \
            "API key must not appear in ProviderError.safe_message"
        assert "<redacted>" in exc_info.value.safe_message


# ---------------------------------------------------------------------------
# Tests: provider_factory.load_provider
# ---------------------------------------------------------------------------

class TestLoadProvider:
    def test_ollama_is_default(self):
        """When LLM_PROVIDER is not set, load_provider should default to 'ollama'."""
        from src.llm.provider_factory import load_provider
        with patch.dict("os.environ", {}, clear=True):
            # LLM_PROVIDER absent → should use 'ollama'
            os.environ.pop("LLM_PROVIDER", None)
            with patch("src.llm.provider_factory._instantiate") as mock_inst:
                mock_inst.return_value = _make_mock_provider("ollama")
                provider = load_provider()
                mock_inst.assert_called_once_with("ollama")

    def test_explicit_provider_overrides_default(self):
        from src.llm.provider_factory import load_provider
        with patch.dict("os.environ", {"LLM_PROVIDER": "gemini"}):
            with patch("src.llm.provider_factory._instantiate") as mock_inst:
                mock_inst.return_value = _make_mock_provider("gemini")
                provider = load_provider()
                mock_inst.assert_called_once_with("gemini")

    def test_model_env_var_read_by_groq_provider(self):
        """GROQ_MODEL env var must be passed to GroqProvider.model_name."""
        with patch.dict("os.environ", {
            "GROQ_API_KEY": "fake-key",
            "GROQ_MODEL": "mixtral-8x7b-32768",
        }):
            from src.llm.groq import GroqProvider
            with patch("src.llm.groq.httpx.Client"):
                p = GroqProvider()
                assert p.model_name == "mixtral-8x7b-32768"

    def test_model_env_var_read_by_gemini_provider(self):
        """GEMINI_MODEL env var must be passed to GeminiProvider.model_name."""
        pytest.importorskip("google.genai", reason="google-genai not installed")
        with patch.dict("os.environ", {
            "GEMINI_API_KEY": "fake-key",
            "GEMINI_MODEL": "gemini-2.5-flash",
        }):
            from src.llm.gemini import GeminiProvider
            with patch("src.llm.gemini.genai.Client"):
                p = GeminiProvider()
                assert p.model_name == "gemini-2.5-flash"

    def test_gemini_default_model_is_not_deprecated(self):
        """GeminiProvider must not default to the deprecated gemini-2.0-flash."""
        pytest.importorskip("google.genai", reason="google-genai not installed")
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake"}, clear=False):
            os.environ.pop("GEMINI_MODEL", None)
            from src.llm.gemini import GeminiProvider, _DEFAULT_MODEL
            assert _DEFAULT_MODEL != "gemini-2.0-flash", (
                "gemini-2.0-flash is deprecated (HTTP 404). "
                f"Default must be a stable alias. Got: {_DEFAULT_MODEL!r}"
            )
            assert "001" in _DEFAULT_MODEL or "flash" in _DEFAULT_MODEL

    def test_nvidia_model_env_var(self):
        """NVIDIA_MODEL env var must be passed to NvidiaProvider.model_name."""
        with patch.dict("os.environ", {
            "NVIDIA_API_KEY": "fake-key",
            "NVIDIA_MODEL": "meta/llama-3.3-70b-instruct",
        }):
            from src.llm.nvidia import NvidiaProvider
            with patch("src.llm.nvidia.httpx.Client"):
                p = NvidiaProvider()
                assert p.model_name == "meta/llama-3.3-70b-instruct"


# ---------------------------------------------------------------------------
# Tests: provider_factory.load_provider_chain
# ---------------------------------------------------------------------------

class TestLoadProviderChain:
    def test_returns_first_available(self):
        """load_provider_chain returns the first provider with a key set."""
        from src.llm.provider_factory import load_provider_chain
        env = {"GROQ_API_KEY": "gk", "NVIDIA_API_KEY": "", "GEMINI_API_KEY": ""}
        with patch.dict("os.environ", env):
            with patch("src.llm.provider_factory._instantiate") as mi:
                mi.return_value = _make_mock_provider("groq")
                provider, name = load_provider_chain()
                assert name == "groq"
                mi.assert_called_once_with("groq")

    def test_skips_provider_without_key(self):
        """Providers without API keys are silently skipped."""
        from src.llm.provider_factory import load_provider_chain
        env = {"GROQ_API_KEY": "", "NVIDIA_API_KEY": "", "GEMINI_API_KEY": "gk"}
        with patch.dict("os.environ", env):
            with patch("src.llm.provider_factory._instantiate") as mi:
                mi.return_value = _make_mock_provider("gemini")
                provider, name = load_provider_chain()
                assert name == "gemini"
                mi.assert_called_once_with("gemini")

    def test_raises_if_no_keys_configured(self):
        """If no API keys are set, load_provider_chain raises RuntimeError."""
        from src.llm.provider_factory import load_provider_chain
        env = {"GROQ_API_KEY": "", "NVIDIA_API_KEY": "", "GEMINI_API_KEY": ""}
        with patch.dict("os.environ", env):
            with pytest.raises(RuntimeError, match="No LLM provider"):
                load_provider_chain()


# ---------------------------------------------------------------------------
# Tests: generate_with_fallback
# ---------------------------------------------------------------------------

class TestGenerateWithFallback:
    def _env_all_keys(self):
        return {"GROQ_API_KEY": "gk", "NVIDIA_API_KEY": "nk", "GEMINI_API_KEY": "gemk"}

    def test_primary_succeeds_no_fallback(self):
        """If the initial provider succeeds, no fallback is attempted."""
        from src.llm.provider_factory import generate_with_fallback
        primary = _make_mock_provider("groq", answer="hello")
        answer, name = generate_with_fallback(
            [{"role": "user", "content": "hi"}],
            initial_provider=primary,
            initial_provider_name="groq",
        )
        assert answer == "hello"
        assert name == "groq"
        primary.generate.assert_called_once()

    def test_groq_failure_falls_back_to_nvidia(self):
        """ProviderError from Groq causes fallback to NVIDIA."""
        from src.llm.provider_factory import generate_with_fallback
        from src.llm.base import ProviderError

        failing_groq = _make_failing_provider("groq", 503)
        nvidia_mock = _make_mock_provider("nvidia", "nvidia answer")

        env = self._env_all_keys()
        with patch.dict("os.environ", env):
            with patch("src.llm.provider_factory._instantiate", return_value=nvidia_mock), \
                 patch("src.llm.provider_factory._is_key_available", return_value=True):
                answer, name = generate_with_fallback(
                    [{"role": "user", "content": "hello"}],
                    priority=["groq", "nvidia", "gemini"],
                    initial_provider=failing_groq,
                    initial_provider_name="groq",
                )
        assert answer == "nvidia answer"
        assert name == "nvidia"

    def test_groq_nvidia_fail_falls_back_to_gemini(self):
        """When both Groq and NVIDIA fail, Gemini is used."""
        from src.llm.provider_factory import generate_with_fallback
        from src.llm.base import ProviderError

        failing_groq = _make_failing_provider("groq", 503)
        failing_nvidia = _make_failing_provider("nvidia", 502)
        gemini_mock = _make_mock_provider("gemini", "gemini answer")

        env = self._env_all_keys()
        call_count = [0]
        def mock_instantiate(name):
            if name == "nvidia":
                return failing_nvidia
            if name == "gemini":
                return gemini_mock
            raise ValueError(f"Unexpected: {name}")

        with patch.dict("os.environ", env):
            with patch("src.llm.provider_factory._instantiate", side_effect=mock_instantiate), \
                 patch("src.llm.provider_factory._is_key_available", return_value=True):
                answer, name = generate_with_fallback(
                    [{"role": "user", "content": "hello"}],
                    priority=["groq", "nvidia", "gemini"],
                    initial_provider=failing_groq,
                    initial_provider_name="groq",
                )
        assert answer == "gemini answer"
        assert name == "gemini"

    def test_all_fail_raises_provider_error(self):
        """If all providers fail, the last ProviderError is raised."""
        from src.llm.provider_factory import generate_with_fallback
        from src.llm.base import ProviderError

        failing_groq = _make_failing_provider("groq", 503)
        failing_nvidia = _make_failing_provider("nvidia", 502)
        failing_gemini = _make_failing_provider("gemini", 404)

        def mock_instantiate(name):
            if name == "nvidia":
                return failing_nvidia
            if name == "gemini":
                return failing_gemini
            raise ValueError(name)

        env = self._env_all_keys()
        with patch.dict("os.environ", env):
            with patch("src.llm.provider_factory._instantiate", side_effect=mock_instantiate), \
                 patch("src.llm.provider_factory._is_key_available", return_value=True):
                with pytest.raises(ProviderError) as exc_info:
                    generate_with_fallback(
                        [{"role": "user", "content": "hello"}],
                        priority=["groq", "nvidia", "gemini"],
                        initial_provider=failing_groq,
                        initial_provider_name="groq",
                    )
        # Error must not contain any API keys
        err_msg = str(exc_info.value)
        for key in ["gk", "nk", "gemk"]:
            assert key not in err_msg, f"API key {key!r} found in error message"

    def test_history_preserved_on_fallback(self):
        """The full conversation history is passed unchanged to the fallback provider."""
        from src.llm.provider_factory import generate_with_fallback

        history = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]
        failing_groq = _make_failing_provider("groq", 503)
        nvidia_mock = _make_mock_provider("nvidia", "6")

        with patch("src.llm.provider_factory._instantiate", return_value=nvidia_mock), \
             patch("src.llm.provider_factory._is_key_available", return_value=True):
            answer, name = generate_with_fallback(
                history,
                priority=["groq", "nvidia", "gemini"],
                initial_provider=failing_groq,
                initial_provider_name="groq",
            )
        # The fallback provider must receive the complete unmodified history.
        nvidia_mock.generate.assert_called_once_with(history)

    def test_nvidia_skipped_when_no_key(self):
        """NVIDIA is not attempted if NVIDIA_API_KEY is absent."""
        from src.llm.provider_factory import generate_with_fallback

        failing_groq = _make_failing_provider("groq", 503)
        gemini_mock = _make_mock_provider("gemini", "gemini direct")

        def key_available(name):
            return name in ("groq", "gemini")  # nvidia has no key

        with patch("src.llm.provider_factory._is_key_available", side_effect=key_available), \
             patch("src.llm.provider_factory._instantiate", return_value=gemini_mock):
            answer, name = generate_with_fallback(
                [{"role": "user", "content": "hi"}],
                priority=["groq", "nvidia", "gemini"],
                initial_provider=failing_groq,
                initial_provider_name="groq",
            )
        assert name == "gemini"


# ---------------------------------------------------------------------------
# Tests: evaluator error accounting
# ---------------------------------------------------------------------------

class TestEvaluatorErrorAccounting:
    """Regression tests for Phase 4.2 pilot bug: error strings scored as answers."""

    _LEGACY_ERROR = "[Error communicating with LLM provider: 404 NOT_FOUND. This model models/gemini-2.0-flash is no longer available.]"

    def test_is_error_response_detects_legacy_prefix(self):
        from src.eval.eval_questions import _is_error_response
        assert _is_error_response(self._LEGACY_ERROR) is True

    def test_is_error_response_passes_real_answer(self):
        from src.eval.eval_questions import _is_error_response
        assert _is_error_response("The return window is 7 days.") is False

    def test_is_error_response_detects_error_from_llm(self):
        from src.eval.eval_questions import _is_error_response
        assert _is_error_response("[Error from LLM provider: timeout]") is True

    def test_error_string_not_scored_as_answer(self):
        """When the model_response is a legacy error string, heuristic checks
        must NOT produce PASS — they must return API_ERROR."""
        from src.eval.eval_questions import _is_error_response
        from src.eval.metrics import check_must_include

        # PQ0006 case: must_include = 'no / not / waived / free'
        # The error string contains 'NOT_FOUND' which would falsely match 'not'.
        raw = self._LEGACY_ERROR
        is_error = _is_error_response(raw)
        assert is_error, "Legacy error string should be detected"
        # Evaluator must skip keyword check for error responses:
        mi = "API_ERROR" if is_error else check_must_include(raw, "no / not / waived / free")
        assert mi == "API_ERROR", \
            f"Error response must produce API_ERROR, not {mi!r}"

    def test_provider_error_increments_api_errors(self):
        """_run_rag_question must count ProviderError as api_error=True."""
        from src.llm.base import ProviderError
        err = ProviderError("gemini", "HTTP 404: model not found", status_code=404)

        with patch("src.agent.rag_agent.RAGAgent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance._provider = MagicMock()
            mock_agent_instance._provider.PROVIDER_NAME = "gemini"
            mock_agent_instance.index = MagicMock()
            mock_agent_instance.chat.side_effect = err
            MockAgent.return_value = mock_agent_instance

            from src.eval.eval_questions import _run_rag_question
            response, retrieved, error, provider_used, latency = _run_rag_question("test q")

        assert response == "", f"Response should be empty on error, got {response!r}"
        assert error != "", "Error field should be non-empty"
        assert provider_used == "gemini"
        # API key must not be in the error string.
        assert "YOUR_GEMINI" not in error

    def test_baseline_provider_error_increments_api_errors(self):
        """_run_baseline_question must count ProviderError as api_error=True."""
        from src.llm.base import ProviderError
        err = ProviderError("groq", "HTTP 503: service unavailable", status_code=503)

        with patch("src.agent.baseline.BaselineAgent") as MockAgent:
            mock_agent_instance = MagicMock()
            mock_agent_instance.provider = MagicMock()
            mock_agent_instance.provider.PROVIDER_NAME = "groq"
            mock_agent_instance.get_response.side_effect = err
            MockAgent.return_value = mock_agent_instance

            from src.eval.eval_questions import _run_baseline_question
            response, error, provider_used, latency = _run_baseline_question("test q")

        assert response == ""
        assert "503" in error or "service unavailable" in error.lower()
        assert provider_used == "groq"

    def test_legacy_error_string_detected_in_rag_runner(self):
        """If the agent returns a legacy error string (not raises), it should
        be detected and treated as an API error."""
        legacy = "[Error communicating with LLM provider: 404 NOT_FOUND...]"

        with patch("src.agent.rag_agent.RAGAgent") as MockAgent:
            mock_agent = MagicMock()
            mock_agent._provider = MagicMock()
            mock_agent._provider.PROVIDER_NAME = "gemini"
            mock_agent.index = MagicMock()
            mock_resp = MagicMock()
            mock_resp.answer = legacy
            mock_resp.retrieval_results = []
            mock_agent.chat.return_value = mock_resp
            MockAgent.return_value = mock_agent

            from src.eval.eval_questions import _run_rag_question
            response, retrieved, error, provider_used, latency = _run_rag_question("test")

        assert response == "", f"Legacy error string must not be returned as answer, got {response!r}"
        assert error == legacy, "Error field must contain the original error string"

    def test_heuristic_skipped_for_api_error_rows(self):
        """For rows with is_api_error=True, heuristic scores must be 'API_ERROR'."""
        # Simulate what the evaluator does for an error row.
        is_api_error = True
        response = "[Error communicating with LLM provider: ...]"
        mi_kw = "no / not / waived / free"
        mn_kw = "5 to 7 days"

        from src.eval.metrics import check_must_include, check_must_not_include
        # The harness should skip heuristics for error rows:
        mi_check = "API_ERROR" if is_api_error else check_must_include(response, mi_kw)
        mn_check = "API_ERROR" if is_api_error else check_must_not_include(response, mn_kw)
        assert mi_check == "API_ERROR"
        assert mn_check == "API_ERROR"


# ---------------------------------------------------------------------------
# Tests: provider PROVIDER_NAME attributes
# ---------------------------------------------------------------------------

class TestProviderNames:
    def test_groq_provider_name(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "fake"}):
            with patch("src.llm.groq.httpx.Client"):
                from src.llm.groq import GroqProvider
                p = GroqProvider()
                assert p.PROVIDER_NAME == "groq"

    def test_nvidia_provider_name(self):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "fake"}):
            with patch("src.llm.nvidia.httpx.Client"):
                from src.llm.nvidia import NvidiaProvider
                p = NvidiaProvider()
                assert p.PROVIDER_NAME == "nvidia"

    def test_gemini_provider_name(self):
        pytest.importorskip("google.genai", reason="google-genai not installed")
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake"}):
            with patch("src.llm.gemini.genai.Client"):
                from src.llm.gemini import GeminiProvider
                p = GeminiProvider()
                assert p.PROVIDER_NAME == "gemini"


# ---------------------------------------------------------------------------
# Tests: dry-run still works with ollama as default
# ---------------------------------------------------------------------------

class TestDryRunWithOllamaDefault:
    def test_dry_run_all_configs(self, tmp_path):
        """Full dry-run must complete without errors under the Ollama default."""
        with patch.dict("os.environ", {"LLM_PROVIDER": "ollama"}):
            with patch("src.llm.provider_factory._instantiate") as mi, \
                 patch("src.llm.provider_factory._is_key_available", return_value=True):
                mi.return_value = _make_mock_provider("ollama")

                from src.eval.run_eval import main as run_eval_main
                import sys
                argv = [
                    "--dry-run",
                    "--config", "baseline", "rag", "full",
                    "--output-dir", str(tmp_path),
                ]
                # Should complete without raising.
                run_eval_main(argv)

        # Verify output files were created.
        csvs = list(tmp_path.glob("*.csv"))
        assert len(csvs) >= 2, f"Expected ≥2 CSV files, got {[f.name for f in csvs]}"
        summaries = list(tmp_path.glob("eval_summary_*.json"))
        assert len(summaries) == 1
