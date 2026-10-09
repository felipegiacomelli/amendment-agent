import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from anthropic import Anthropic

from amendment_agent import evals
from amendment_agent.agent import StopReason
from amendment_agent.evals import CaseResult, format_table, run_eval
from amendment_agent.graders import Grade
from amendment_agent.recorder import RunRecorder
from amendment_agent.schema import EvalCase

type FakeAgent = Callable[..., StopReason]
type FakeGrader = Callable[..., Grade]

MODEL = "claude-sonnet-5-5"
CLIENT = Anthropic(api_key="test-key-not-used")
CASE = EvalCase.model_validate(
    {
        "id": "case-1",
        "category": "superseded",
        "question": "Q?",
        "documents": ["original"],
        "expected_status": "answered",
        "accepted_values": ["3.75:1"],
        "expected_citations": [{"document": "original", "section": "7.06(a)"}],
    }
)
GOOD = json.dumps(
    {
        "status": "answered",
        "value": "3.75:1",
        "explanation": "x",
        "citations": [{"document": "original", "section": "7.06(a)", "lines": [1, 2]}],
    }
)


def agent_writing(content: str | None) -> FakeAgent:
    """A fake run_agent that writes `content` as answer.json (or nothing) and stops."""

    def fake(
        question: str, *, recorder: RunRecorder, output_dir: Path, **_: object
    ) -> StopReason:
        if content is not None:
            output_dir.mkdir(parents=True)
            (output_dir / "answer.json").write_text(content, encoding="utf-8")
        recorder.end("answered", [])
        return StopReason.ANSWERED

    return fake


def passing(*_: object) -> Grade:
    return Grade(passed=True, reason="ok")


def run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent: FakeAgent,
    cases: list[EvalCase] | None = None,
    answer_grader: FakeGrader = passing,
    citation_grader: FakeGrader = passing,
) -> list[CaseResult]:
    monkeypatch.setattr(evals, "run_agent", agent)
    monkeypatch.setattr(evals, "grade_answer", answer_grader)
    monkeypatch.setattr(evals, "grade_citations", citation_grader)
    cases = cases or [CASE]
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        "".join(c.model_dump_json() + "\n" for c in cases), encoding="utf-8"
    )
    return run_eval(
        cases,
        client=CLIENT,
        model=MODEL,
        max_turns=5,
        max_cost_usd=1.0,
        config={"max_turns": 5},
        cases_path=cases_path,
        runs_dir=tmp_path / "runs",
        outputs_dir=tmp_path / "outputs",
        results_dir=tmp_path / "results",
        text_dir=tmp_path / "text",
    )


def test_passing_case_links_transcript_output_and_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    [result] = run(tmp_path, monkeypatch, agent_writing(GOOD))
    assert result.answer.passed and result.citations.passed
    assert result.stop_reason == "answered" and result.category == "superseded"
    assert (tmp_path / "runs" / f"{result.run_id}.jsonl").exists()
    assert (tmp_path / "outputs" / result.run_id / "answer.json").exists()
    [results_file] = list((tmp_path / "results").glob("*.json"))
    payload = json.loads(results_file.read_text(encoding="utf-8"))
    cases_sha = hashlib.sha256((tmp_path / "cases.jsonl").read_bytes()).hexdigest()
    assert payload["cases_sha256"] == cases_sha and payload["model"] == MODEL
    assert payload["results"][0]["run_id"] == result.run_id
    assert set(payload["git"]) == {"commit", "dirty"}
    run_start = json.loads(
        (tmp_path / "runs" / f"{result.run_id}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert run_start["config"]["cases_sha256"] == cases_sha
    assert run_start["config"]["eval_id"] == payload["eval_id"]
    assert payload["grading_sha256"] == evals.grading_fingerprint()
    assert len(payload["grading_sha256"]) == 64
    assert run_start["config"]["grading_sha256"] == payload["grading_sha256"]


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (None, "no answer file"),
        ("{not json", "invalid answer file"),
        (
            json.dumps({"status": "answered", "value": "x", "explanation": "x"}),
            "invalid answer file",
        ),
    ],
)
def test_missing_or_invalid_answer_fails_both_grades_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str | None, reason: str
) -> None:
    [result] = run(tmp_path, monkeypatch, agent_writing(content))
    assert not result.answer.passed and not result.citations.passed
    assert reason in result.answer.reason


def test_agent_exception_fails_the_case_and_the_eval_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    good = agent_writing(GOOD)

    def flaky(question: str, **kwargs: Any) -> StopReason:
        calls.append(question)
        if len(calls) == 1:
            raise RuntimeError("loop crashed")
        return good(question, **kwargs)

    second = CASE.model_copy(update={"id": "case-2"})
    first_result, second_result = run(
        tmp_path, monkeypatch, flaky, cases=[CASE, second]
    )
    assert (
        not first_result.answer.passed
        and "RuntimeError: loop crashed" in first_result.answer.reason
    )
    assert first_result.stop_reason == "error"
    assert second_result.answer.passed


def test_unreadable_answer_file_fails_the_case_and_the_eval_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_read_text = Path.read_text
    denied: list[Path] = []

    def read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.name == "answer.json" and not denied:  # the first case's file
            denied.append(self)
            raise PermissionError("denied")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    second = CASE.model_copy(update={"id": "case-2"})
    first_result, second_result = run(
        tmp_path, monkeypatch, agent_writing(GOOD), cases=[CASE, second]
    )
    assert not first_result.answer.passed and not first_result.citations.passed
    assert "unreadable answer file" in first_result.answer.reason
    assert second_result.answer.passed and second_result.citations.passed
    assert len(list((tmp_path / "results").glob("*.json"))) == 1


def test_grader_exception_fails_only_that_grade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: object) -> Grade:
        raise KeyError("missing field")

    [result] = run(tmp_path, monkeypatch, agent_writing(GOOD), answer_grader=broken)
    assert not result.answer.passed and "KeyError" in result.answer.reason
    assert result.citations.passed


def test_not_implemented_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unfinished(*_: object, **__: object) -> StopReason:
        raise NotImplementedError

    with pytest.raises(NotImplementedError):
        run(tmp_path, monkeypatch, unfinished)


def test_table_shows_marks_totals_reasons_and_incomplete_cost() -> None:
    ok, bad = Grade(True, "ok"), Grade(False, "wrong section")
    table = format_table(
        [CaseResult("case-1", "superseded", ok, bad, "answered", 3, 0.02, False, "r1")]
    )
    assert "case-1" in table and "superseded" in table and "✓" in table and "✗" in table
    assert "answer 1/1" in table and "citations 0/1" in table
    assert "wrong section" in table and "0.0200?" in table
