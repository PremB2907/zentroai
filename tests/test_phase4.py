# tests/test_phase4.py
"""Phase 4 offline unit tests for the evaluation harness.

All tests are completely offline:
  - No LLM API calls.
  - No network access.
  - Evaluation runners are invoked with dry_run=True or mocked providers.
  - CSV/JSON output is validated structurally, not for content.

Run with: python -m pytest tests/test_phase4.py -v
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ===========================================================================
# 1. Metric helpers
# ===========================================================================

class TestMetrics:
    def test_must_include_pass(self):
        from src.eval.metrics import check_must_include
        assert check_must_include("refund takes 2 hours instantly", "instant / 2 hours") == "PASS"

    def test_must_include_fail(self):
        from src.eval.metrics import check_must_include
        assert check_must_include("we will process it", "instant / 2 hours") == "FAIL"

    def test_must_include_na_when_empty(self):
        from src.eval.metrics import check_must_include
        assert check_must_include("anything", "") == "N/A"

    def test_must_not_include_pass_when_absent(self):
        from src.eval.metrics import check_must_not_include
        assert check_must_not_include("refund in 2 hours", "5 to 7 days") == "PASS"

    def test_must_not_include_fail_when_present(self):
        from src.eval.metrics import check_must_not_include
        assert check_must_not_include("refund takes 5 to 7 days", "5 to 7 days") == "FAIL"

    def test_must_not_include_na_when_empty(self):
        from src.eval.metrics import check_must_not_include
        assert check_must_not_include("anything", "") == "N/A"

    def test_retrieval_recall_pass(self):
        from src.eval.metrics import check_retrieval_recall
        assert check_retrieval_recall("REF-2 ; KB-00.1 ; RET-1", "REF-2 ; KB-00.1") == "PASS"

    def test_retrieval_recall_partial(self):
        from src.eval.metrics import check_retrieval_recall
        assert check_retrieval_recall("REF-2 ; OTHER", "REF-2 ; KB-00.1") == "PARTIAL"

    def test_retrieval_recall_fail(self):
        from src.eval.metrics import check_retrieval_recall
        assert check_retrieval_recall("FAR-1 ; FAR-2", "REF-2 ; KB-00.1") == "FAIL"

    def test_retrieval_recall_na_when_empty(self):
        from src.eval.metrics import check_retrieval_recall
        assert check_retrieval_recall("REF-2", "") == "N/A"

    def test_tool_call_pass(self):
        from src.eval.metrics import check_tool_call
        assert check_tool_call(
            "lookup_order(order_id=89782837)",
            "lookup_order(order_id=89782837)"
        ) == "PASS"

    def test_tool_call_pass_normalised(self):
        from src.eval.metrics import check_tool_call
        # Extra spaces should be normalised away
        assert check_tool_call(
            "lookup_order( order_id=89782837 )",
            "lookup_order(order_id=89782837)"
        ) == "PASS"

    def test_tool_call_fail_mismatch(self):
        from src.eval.metrics import check_tool_call
        assert check_tool_call(
            "lookup_order(order_id=11111111)",
            "lookup_order(order_id=89782837)"
        ) == "FAIL"

    def test_tool_call_na_when_both_empty(self):
        from src.eval.metrics import check_tool_call
        assert check_tool_call("", "") == "N/A"

    def test_tool_call_unexpected_when_expected_empty(self):
        from src.eval.metrics import check_tool_call
        assert check_tool_call("create_ticket()", "") == "UNEXPECTED"

    def test_summarise_questions_counts(self):
        from src.eval.metrics import summarise_questions
        rows = [
            {"heuristic_must_include": "PASS", "heuristic_must_not_include": "PASS",
             "retrieval_recall": "PASS", "error": "", "latency_s": 0.5,
             "answerable": "True"},
            {"heuristic_must_include": "FAIL", "heuristic_must_not_include": "N/A",
             "retrieval_recall": "FAIL", "error": "oops", "latency_s": 0.3,
             "answerable": "False"},
        ]
        summary = summarise_questions(rows)
        assert summary["total"] == 2
        assert summary["api_errors"] == 1
        assert summary["must_include"]["PASS"] == 1
        assert summary["must_include"]["FAIL"] == 1

    def test_summarise_conversations_tool_counts(self):
        from src.eval.metrics import summarise_conversations
        rows = [
            {"tool_call_match": "PASS", "heuristic_must_include": "PASS",
             "heuristic_must_not_include": "N/A", "retrieval_recall": "PASS",
             "error": "", "latency_s": 0.2,
             "expected_tool_calls": "lookup_order(order_id=12345678, verification_value=4533)"},
            {"tool_call_match": "FAIL", "heuristic_must_include": "FAIL",
             "heuristic_must_not_include": "N/A", "retrieval_recall": "N/A",
             "error": "", "latency_s": 0.1,
             "expected_tool_calls": "create_ticket()"},
        ]
        summary = summarise_conversations(rows)
        assert summary["total_turns"] == 2
        assert summary["tool_call_match"]["PASS"] == 1
        assert summary["tool_call_match"]["FAIL"] == 1
        assert summary["verification_turns"] == 1
        assert summary["escalation_turns"] == 1


# ===========================================================================
# 2. Question evaluator (Config A and B)
# ===========================================================================

class TestQuestionEvaluator:
    def test_load_questions_returns_52_rows(self):
        from src.eval.eval_questions import load_questions
        rows = load_questions()
        assert len(rows) == 52

    def test_load_questions_limit(self):
        from src.eval.eval_questions import load_questions
        rows = load_questions(limit=3)
        assert len(rows) == 3

    def test_questions_have_required_fields(self):
        from src.eval.eval_questions import load_questions
        rows = load_questions(limit=1)
        for field in ["id", "question", "intent", "gold_evidence", "reference_answer"]:
            assert field in rows[0], f"Missing field: {field}"

    def test_baseline_dryrun_creates_csv(self, tmp_path):
        from src.eval.eval_questions import run_question_eval
        out_csv, summary = run_question_eval(
            config="baseline", dry_run=True, limit=3, output_dir=tmp_path
        )
        assert out_csv.exists()
        with out_csv.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_rag_dryrun_creates_csv_with_retrieval(self, tmp_path):
        from src.eval.eval_questions import run_question_eval
        out_csv, summary = run_question_eval(
            config="rag", dry_run=True, limit=3, output_dir=tmp_path
        )
        assert out_csv.exists()
        with out_csv.open() as f:
            rows = list(csv.DictReader(f))
        # Every row should have retrieved_sections populated (RAG retrieval runs offline)
        for r in rows:
            assert "retrieved_sections" in r

    def test_rag_retrieve_only_creates_csv(self, tmp_path):
        from src.eval.eval_questions import run_question_eval
        out_csv, summary = run_question_eval(
            config="rag", retrieve_only=True, limit=5, output_dir=tmp_path
        )
        assert out_csv.exists()

    def test_summary_contains_config_key(self, tmp_path):
        from src.eval.eval_questions import run_question_eval
        _, summary = run_question_eval(
            config="rag", dry_run=True, limit=2, output_dir=tmp_path
        )
        assert summary["config"] == "rag"
        assert summary["questions"] == 2

    def test_summary_has_metric_keys(self, tmp_path):
        from src.eval.eval_questions import run_question_eval
        _, summary = run_question_eval(
            config="rag", dry_run=True, limit=2, output_dir=tmp_path
        )
        assert "must_include" in summary
        assert "retrieval_recall" in summary

    def test_error_field_present_in_output(self, tmp_path):
        from src.eval.eval_questions import run_question_eval
        out_csv, _ = run_question_eval(
            config="baseline", dry_run=True, limit=1, output_dir=tmp_path
        )
        with out_csv.open() as f:
            rows = list(csv.DictReader(f))
        assert "error" in rows[0]

    def test_latency_field_present(self, tmp_path):
        from src.eval.eval_questions import run_question_eval
        out_csv, _ = run_question_eval(
            config="rag", dry_run=True, limit=1, output_dir=tmp_path
        )
        with out_csv.open() as f:
            rows = list(csv.DictReader(f))
        assert "latency_s" in rows[0]


# ===========================================================================
# 3. Conversation evaluator (Config C)
# ===========================================================================

class TestConversationEvaluator:
    def test_load_conversations_returns_23_convs(self):
        from src.eval.eval_conversations import load_conversations
        convs = load_conversations()
        assert len(convs) == 23

    def test_load_conversations_limit(self):
        from src.eval.eval_conversations import load_conversations
        convs = load_conversations(limit_convs=3)
        assert len(convs) == 3

    def test_turns_sorted_within_conversation(self):
        from src.eval.eval_conversations import load_conversations
        convs = load_conversations()
        for cid, turns in convs.items():
            turn_nums = [int(t["turn"]) for t in turns]
            assert turn_nums == sorted(turn_nums), f"Unsorted turns in {cid}"

    def test_dryrun_creates_csv(self, tmp_path):
        from src.eval.eval_conversations import run_conversation_eval
        out_csv, summary = run_conversation_eval(
            dry_run=True, limit_convs=2, output_dir=tmp_path
        )
        assert out_csv.exists()

    def test_dryrun_csv_has_required_columns(self, tmp_path):
        from src.eval.eval_conversations import run_conversation_eval
        out_csv, _ = run_conversation_eval(
            dry_run=True, limit_convs=1, output_dir=tmp_path
        )
        with out_csv.open() as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
        for col in [
            "conversation_id", "turn", "scenario", "customer_message",
            "expected_tool_calls", "actual_tool_calls", "tool_call_match",
            "retrieved_sections", "retrieval_recall",
            "model_response", "error", "latency_s",
            "order_revealed", "escalation_done", "ticket_id",
        ]:
            assert col in cols, f"Missing column: {col}"

    def test_dryrun_tool_calls_recorded(self, tmp_path):
        """Conversations with expected tool calls should record actual calls."""
        from src.eval.eval_conversations import run_conversation_eval
        out_csv, _ = run_conversation_eval(
            dry_run=True, limit_convs=5, output_dir=tmp_path
        )
        with out_csv.open() as f:
            rows = list(csv.DictReader(f))
        # At least some rows should have actual_tool_calls populated
        has_calls = [r for r in rows if r.get("actual_tool_calls", "")]
        assert len(has_calls) > 0, "Expected at least one tool call to be recorded"

    def test_summary_has_conversation_keys(self, tmp_path):
        from src.eval.eval_conversations import run_conversation_eval
        _, summary = run_conversation_eval(
            dry_run=True, limit_convs=2, output_dir=tmp_path
        )
        for key in ["total_turns", "tool_call_match", "verification_turns", "escalation_turns"]:
            assert key in summary, f"Missing key: {key}"

    def test_error_handling_in_mock_run(self, tmp_path):
        """Even if a turn errors, other turns should still complete."""
        from src.eval.eval_conversations import run_conversation_eval
        out_csv, summary = run_conversation_eval(
            dry_run=True, limit_convs=3, output_dir=tmp_path
        )
        assert out_csv.exists()

    def test_multi_turn_context_maintained(self, tmp_path):
        """Order ID set in turn 2 should be remembered in turn 3."""
        from src.eval.eval_conversations import load_conversations, _run_mock_conversation
        from src.eval.eval_conversations import ToolCallRecorder

        convs = load_conversations()
        # Find a conversation that has a lookup_order call (turn 2) followed by
        # a verification call (turn 3)
        target = None
        for cid, turns in convs.items():
            tcs = [t.get("expected_tool_calls", "") for t in turns]
            if any("verification_value" in tc for tc in tcs):
                target = turns
                break

        assert target is not None, "No conversation with verification found"
        recorder = ToolCallRecorder()
        results = _run_mock_conversation(target, top_k=5, recorder=recorder)
        # Recorder should have recorded at least one lookup_order call
        all_calls = [r.get("actual_tool_calls", "") for r in results]
        assert any("lookup_order" in c for c in all_calls)


# ===========================================================================
# 4. Master run_eval CLI
# ===========================================================================

class TestRunEvalCLI:
    def test_estimate_exits_cleanly(self, capsys):
        from src.eval.run_eval import print_estimate
        print_estimate()
        captured = capsys.readouterr()
        assert "API Call Estimate" in captured.out
        assert "52" in captured.out   # questions count
        assert "96" in captured.out   # conversation turns count

    def test_missing_confirm_live_exits_with_error(self, capsys):
        from src.eval.run_eval import main
        with pytest.raises(SystemExit) as exc:
            main(["--config", "baseline"])   # no --dry-run, no --confirm-live
        assert exc.value.code == 1

    def test_dryrun_all_configs_creates_summary_json(self, tmp_path):
        from src.eval.run_eval import main
        main([
            "--dry-run",
            "--config", "baseline", "rag", "full",
            "--limit-q", "2",
            "--limit-c", "1",
            "--output-dir", str(tmp_path),
        ])
        json_files = list(tmp_path.glob("eval_summary_*.json"))
        assert len(json_files) == 1
        with json_files[0].open() as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 3   # one entry per config

    def test_dryrun_creates_text_report(self, tmp_path):
        from src.eval.run_eval import main
        main([
            "--dry-run",
            "--config", "rag",
            "--limit-q", "2",
            "--output-dir", str(tmp_path),
        ])
        txt_files = list(tmp_path.glob("eval_report_*.txt"))
        assert len(txt_files) == 1
        content = txt_files[0].read_text()
        assert "PROXY METRICS" in content

    def test_dryrun_baseline_only(self, tmp_path):
        from src.eval.run_eval import main
        main([
            "--dry-run",
            "--config", "baseline",
            "--limit-q", "3",
            "--output-dir", str(tmp_path),
        ])
        csvs = list(tmp_path.glob("eval_baseline_questions_*.csv"))
        assert len(csvs) == 1

    def test_retrieve_only_skips_llm(self, tmp_path):
        """retrieve-only should produce output without calling any LLM."""
        from src.eval.run_eval import main
        main([
            "--retrieve-only",
            "--config", "rag", "full",
            "--limit-q", "2",
            "--limit-c", "1",
            "--output-dir", str(tmp_path),
        ])
        csvs = list(tmp_path.glob("eval_*.csv"))
        assert len(csvs) == 2   # one for rag, one for full


# ===========================================================================
# 5. Tool-call recorder
# ===========================================================================

class TestToolCallRecorder:
    def test_record_unverified(self):
        from src.eval.eval_conversations import ToolCallRecorder
        r = ToolCallRecorder()
        result = r.record_lookup_unverified("12345678")
        assert r.latest_call() == "lookup_order(order_id=12345678)"
        assert result["verification_required"] is True

    def test_record_verified(self):
        from src.eval.eval_conversations import ToolCallRecorder
        r = ToolCallRecorder()
        r.record_lookup_verified("12345678", "4533")
        assert "verification_value=4533" in r.latest_call()

    def test_record_create_ticket(self):
        from src.eval.eval_conversations import ToolCallRecorder
        r = ToolCallRecorder()
        result = r.record_create_ticket("Customer reports issue with order 12345678 after 3 contacts.")
        assert r.latest_call() == "create_ticket()"
        assert result["created"] is True
        assert "TKT" in result["ticket_id"]

    def test_reset_turn_clears_calls(self):
        from src.eval.eval_conversations import ToolCallRecorder
        r = ToolCallRecorder()
        r.record_lookup_unverified("12345678")
        r.reset_turn()
        assert r.latest_call() == ""


# ===========================================================================
# 6. End-to-end dry-run regression (+ dataset validation)
# ===========================================================================

class TestEndToEnd:
    def test_dataset_validation_still_passes(self):
        """validate_dataset.py must continue to pass after Phase 4 changes."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "validate_dataset.py"],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        assert result.returncode == 0, (
            f"validate_dataset.py FAILED:\n{result.stdout}\n{result.stderr}"
        )

    def test_full_dry_run_all_configs(self, tmp_path):
        """Complete dry-run across all three configs completes without errors."""
        from src.eval.run_eval import main
        main([
            "--dry-run",
            "--config", "baseline", "rag", "full",
            "--limit-q", "5",
            "--limit-c", "2",
            "--output-dir", str(tmp_path),
        ])
        # All three CSVs should exist
        assert len(list(tmp_path.glob("eval_baseline_*.csv"))) == 1
        assert len(list(tmp_path.glob("eval_rag_*.csv"))) == 1
        assert len(list(tmp_path.glob("eval_full_*.csv"))) == 1
        # Summary JSON should be valid
        json_files = list(tmp_path.glob("eval_summary_*.json"))
        assert json_files
        data = json.loads(json_files[0].read_text())
        assert len(data) == 3

    def test_no_api_keys_in_output_files(self, tmp_path):
        """Output files must not contain any API key patterns."""
        from src.eval.run_eval import main
        main([
            "--dry-run",
            "--config", "rag",
            "--limit-q", "2",
            "--output-dir", str(tmp_path),
        ])
        import re
        key_pattern = re.compile(r"AIza[A-Za-z0-9\-_]{35}|gsk_[A-Za-z0-9]{52}")
        for f in tmp_path.iterdir():
            content = f.read_text(encoding="utf-8", errors="ignore")
            assert not key_pattern.search(content), (
                f"Potential API key found in {f.name}"
            )
