import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from amendment_agent.schema import (
    Answer,
    AnswerStatus,
    EvalCase,
    load_answer,
    load_cases,
)
from amendment_agent.settings import CASES_PATH

CITATION = {"document": "amendment-1", "section": "7.06(a)", "lines": [10, 12]}
ANSWER: dict[str, Any] = {
    "status": "answered",
    "value": "not tested",
    "explanation": "Amendment No. 1 suspends the test for that quarter.",
    "citations": [CITATION],
}
CASE: dict[str, Any] = {
    "id": "case-1",
    "category": "superseded",
    "question": "Q?",
    "documents": ["original", "amendment-1"],
    "expected_status": "answered",
    "accepted_values": ["not tested"],
    "expected_citations": [{"document": "amendment-1", "section": "7.06(a)"}],
}


def test_valid_answer_loads(tmp_path: Path) -> None:
    path = tmp_path / "answer.json"
    path.write_text(json.dumps(ANSWER), encoding="utf-8")
    answer = load_answer(path)
    assert answer.status is AnswerStatus.ANSWERED and answer.citations[0].lines == (
        10,
        12,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"value": None},
        {"value": "  "},
        {"citations": []},
        {"citations": [CITATION | {"lines": [12, 10]}]},
        {"citations": [CITATION | {"lines": [0, 3]}]},
        {"status": "maybe"},
        {"extra_field": 1},
    ],
)
def test_invalid_answers_are_rejected(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Answer.model_validate(ANSWER | change)


def test_insufficient_evidence_needs_no_value() -> None:
    answer = Answer.model_validate(
        {"status": "insufficient_evidence", "explanation": "Not in these documents."}
    )
    assert answer.value is None and answer.citations == []


def test_load_answer_names_the_problem(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no answer file"):
        load_answer(tmp_path / "missing.json")
    bad = tmp_path / "answer.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid answer file"):
        load_answer(bad)


@pytest.mark.parametrize(
    "change",
    [
        {"accepted_values": []},
        {"expected_citations": []},
        {"expected_status": "insufficient_evidence"},  # still has accepted_values
        {"category": "other"},
        {"id": "Not A Slug"},
    ],
)
def test_invalid_cases_are_rejected(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EvalCase.model_validate(CASE | change)


def test_committed_cases_load() -> None:
    assert load_cases(CASES_PATH)


def test_load_cases_reports_line_and_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    line = json.dumps(CASE)
    path.write_text(f"{line}\n\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate case ids"):
        load_cases(path)
    path.write_text(f"{line}\n{{}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"cases\.jsonl:2"):
        load_cases(path)
