"""Data contracts at the eval boundary: the agent's answer file and the eval cases."""

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

SLUG = r"^[a-z0-9]+(-[a-z0-9]+)*$"
ANSWER_FILENAME = "answer.json"


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Citation(BaseModel):
    """Evidence location: a document id, a section, and an inclusive 1-based line range
    in data/text/<document>.txt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document: str
    section: str
    lines: tuple[int, int]

    @model_validator(mode="after")
    def check_lines(self) -> Self:
        start, end = self.lines
        if start < 1 or end < start:
            raise ValueError(f"invalid line range {list(self.lines)}")
        return self


class Answer(BaseModel):
    """The agent's final answer, written to outputs/<run_id>/answer.json.

    The file is authoritative: graders read it, nothing else.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: AnswerStatus
    value: str | None = None  # short canonical value, e.g. "3.75:1"; graded
    explanation: str  # short justification; not graded
    citations: list[Citation] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_answered(self) -> Self:
        if self.status is AnswerStatus.ANSWERED:
            if not (self.value and self.value.strip()):
                raise ValueError("an answered status needs a non-empty value")
            if not self.citations:
                raise ValueError("an answered status needs at least one citation")
        return self


def load_answer(path: Path) -> Answer:
    """Parse an answer file. Raise ValueError saying what is wrong (fail closed)."""
    if not path.is_file():
        raise ValueError(f"no answer file at {path}")
    try:
        return Answer.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"unreadable answer file {path}: {error}") from error
    except ValidationError as error:
        raise ValueError(f"invalid answer file {path}: {error}") from error


class Category(StrEnum):
    UNCHANGED = "unchanged"  # no amendment touches the provision
    SUPERSEDED = "superseded"  # replaced by Amendment No. 1's restatement
    LATER_AMENDMENT = "later_amendment"  # changed again by a later amendment
    UNANSWERABLE = "unanswerable"  # the documents don't contain the answer


class ExpectedCitation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document: str
    section: str


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=SLUG)
    category: Category
    question: str
    documents: list[str]  # documents needed to answer; for diagnosis, not graded
    expected_status: AnswerStatus
    accepted_values: list[str] = Field(default_factory=list)
    expected_citations: list[ExpectedCitation] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def check_expectations(self) -> Self:
        if self.expected_status is AnswerStatus.ANSWERED:
            if not self.accepted_values or not self.expected_citations:
                raise ValueError(
                    "an answered case needs accepted_values and expected_citations"
                )
        elif self.accepted_values:
            raise ValueError("an insufficient_evidence case has no accepted_values")
        return self


def load_cases(path: Path) -> list[EvalCase]:
    """One JSON case per line; blank lines skipped; ids unique."""
    cases: list[EvalCase] = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            cases.append(EvalCase.model_validate_json(line))
        except ValidationError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    ids = [case.id for case in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate case ids: {duplicates}")
    return cases
