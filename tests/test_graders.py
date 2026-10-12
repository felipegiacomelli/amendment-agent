from pathlib import Path
from typing import Any

import pytest

from amendment_agent.graders import grade_answer, grade_citations, normalize_value
from amendment_agent.schema import Answer, EvalCase

LEVERAGE = "Permit the Consolidated Leverage Ratio to be greater than 2.50:1"
AMENDED = "Section 7.06(b) of the Credit Agreement is amended to read 2.00:1"


@pytest.fixture
def text_dir(tmp_path: Path) -> Path:
    (tmp_path / "original.txt").write_text(
        "[Page 1]\n"
        f"7.06 Financial Covenants. (a) Leverage. {LEVERAGE}.\n"
        "(b) Interest Coverage. Permit the ratio to be less than 3.00:1.\n",
        encoding="utf-8",
    )
    (tmp_path / "amendment-1.txt").write_text(
        f"[Page 1]\n1.Amendments. {AMENDED} for the quarter ending June 30, 2020.\n",
        encoding="utf-8",
    )
    return tmp_path


def case(**fields: Any) -> EvalCase:
    return EvalCase.model_validate(
        {
            "id": "icr-q2",
            "category": "superseded",
            "question": "Q?",
            "documents": ["amendment-1"],
            "expected_status": "answered",
            "accepted_values": ["2.00:1", "minimum 2.00:1"],
            "expected_citations": [{"document": "amendment-1", "section": "7.06(b)"}],
        }
        | fields
    )


UNANSWERABLE = case(
    category="unanswerable",
    expected_status="insufficient_evidence",
    accepted_values=[],
    expected_citations=[],
)


def cite(
    document: str = "amendment-1",
    section: str = "7.06(b)",
    lines: tuple[int, int] = (2, 2),
    quote: str = AMENDED,
) -> dict[str, Any]:
    return {"document": document, "section": section, "lines": lines, "quote": quote}


def answer(value: str = "2.00:1", *citations: dict[str, Any]) -> Answer:
    return Answer.model_validate(
        {
            "status": "answered",
            "value": value,
            "explanation": "e",
            "citations": list(citations) or [cite()],
        }
    )


INSUFFICIENT = Answer.model_validate(
    {"status": "insufficient_evidence", "explanation": "not in the documents"}
)


@pytest.mark.parametrize(
    ("raw", "normalised"),
    [
        ("3.75:1", "3.75:1"),
        ("3.75 to 1.00", "3.75:1"),
        ("3.75x", "3.75:1"),
        ("3.750 : 1.0", "3.75:1"),
        ("2.00 to 1.00", "2:1"),
        ("Minimum  2.00:1.", "minimum 2:1"),
        ("Not tested", "not tested"),
        ("$20,000,000", "$20,000,000"),
        ("10.00:1", "10:1"),
    ],
)
def test_normalize_value(raw: str, normalised: str) -> None:
    assert normalize_value(raw) == normalised


@pytest.mark.parametrize("value", ["2:1", "2.00 to 1.00", "Minimum 2.00:1."])
def test_an_equivalent_value_passes(value: str) -> None:
    assert grade_answer(case(), answer(value)).passed


@pytest.mark.parametrize(
    "value",
    [
        "3.00:1",  # the right ratio for another period
        "maximum 2.00:1",  # the wrong direction
        "2.00:1 for the quarter ending June 30, 2020",  # not a bare value
    ],
)
def test_a_different_value_fails(value: str) -> None:
    grade = grade_answer(case(), answer(value))
    assert not grade.passed and "not in" in grade.reason


def test_insufficient_evidence_on_an_answerable_case_fails() -> None:
    grade = grade_answer(case(), INSUFFICIENT)
    assert not grade.passed and "expected answered" in grade.reason


def test_an_answer_to_an_unanswerable_case_fails() -> None:
    assert not grade_answer(UNANSWERABLE, answer()).passed


def test_insufficient_evidence_on_an_unanswerable_case_passes(text_dir: Path) -> None:
    assert grade_answer(UNANSWERABLE, INSUFFICIENT).passed
    assert grade_citations(UNANSWERABLE, INSUFFICIENT, text_dir).passed


def test_the_expected_citation_passes_cleanly(text_dir: Path) -> None:
    grade = grade_citations(case(), answer(), text_dir)
    assert grade.passed and grade.warning == ""


def test_a_more_specific_section_passes(text_dir: Path) -> None:
    expected = [{"document": "original", "section": "7.06"}]
    cited = cite("original", "Section 7.06(a)", quote=LEVERAGE)
    assert grade_citations(
        case(expected_citations=expected), answer("x", cited), text_dir
    ).passed


@pytest.mark.parametrize("section", ["7.06", "7.06(a)"])
def test_a_broader_or_sibling_section_fails(text_dir: Path, section: str) -> None:
    grade = grade_citations(case(), answer("2.00:1", cite(section=section)), text_dir)
    assert not grade.passed and "missing: amendment-1 §7.06(b)" in grade.reason


def test_a_missing_expected_citation_fails(text_dir: Path) -> None:
    only_original = cite("original", "7.06", quote=LEVERAGE)
    grade = grade_citations(case(), answer("2.00:1", only_original), text_dir)
    assert not grade.passed
    assert "missing: amendment-1 §7.06(b)" in grade.reason
    assert "extra: original §7.06 [2-2]" in grade.reason


def test_an_extra_citation_passes_with_a_warning(text_dir: Path) -> None:
    superseded = cite("original", "7.06", quote=LEVERAGE)
    grade = grade_citations(case(), answer("2.00:1", cite(), superseded), text_dir)
    assert grade.passed
    assert grade.warning == "extra: original §7.06 [2-2]"


def test_a_citation_to_the_wrong_document_fails(text_dir: Path) -> None:
    wrong = cite("original", "7.06(b)", quote=LEVERAGE)
    assert not grade_citations(case(), answer("2.00:1", wrong), text_dir).passed


@pytest.mark.parametrize(
    ("citation", "problem"),
    [
        (cite(lines=(2, 9)), "past its end"),  # fabricated line range
        (cite(document="amendment-9"), "no document"),
        (cite(quote="a quote that is nowhere in the file"), "quote not found"),
    ],
)
def test_a_citation_the_documents_dont_back_fails(
    text_dir: Path, citation: dict[str, Any], problem: str
) -> None:
    grade = grade_citations(case(), answer("2.00:1", citation), text_dir)
    assert not grade.passed and problem in grade.reason
