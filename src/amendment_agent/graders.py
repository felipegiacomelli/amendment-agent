"""Deterministic eval graders. USER-OWNED: the scaffold defines the contract only.

The eval runner calls these only with a valid `Answer`; a missing or malformed
answer file has already failed. Graders must still fail closed: any doubt is a fail.
"""

import re
from pathlib import Path

import attrs

from amendment_agent.schema import Answer, AnswerStatus, Citation, EvalCase
from amendment_agent.tools import citation_problem

# "3.75:1", "3.75 to 1.00", "3.75x": a number, then ":" or "to" and a number, or "x".
_RATIO = re.compile(r"(\d+(?:\.\d+)?)\s*(?:(?::|\bto\b)\s*(\d+(?:\.\d+)?)|x\b)")
# The section a citation names: "7.06(a)(i)" in "Section 7.06(a)(i)".
_SECTION = re.compile(r"\d+(?:\.\d+)*(?:\([a-z0-9]+\))*")


@attrs.frozen
class Grade:
    passed: bool
    reason: str
    # Set on a pass that still did something worth fixing, e.g. extra citations.
    warning: str = ""


def grade_answer(case: EvalCase, answer: Answer) -> Grade:
    """Status matches `case.expected_status` and, if answered, the value is accepted.

    Proves: the agent's stated value equals one of `case.accepted_values` under
    `normalize_value`. Does not prove: that the explanation is right.
    """
    if answer.status is not case.expected_status:
        return Grade(False, f"status {answer.status}, expected {case.expected_status}")
    if case.expected_status is AnswerStatus.INSUFFICIENT_EVIDENCE:
        return Grade(True, "insufficient_evidence, as expected")
    value = normalize_value(answer.value or "")
    accepted = {normalize_value(v) for v in case.accepted_values}
    if value in accepted:
        return Grade(True, f"value {answer.value!r} accepted")
    return Grade(
        False,
        f"value {answer.value!r} (normalised {value!r}) not in {sorted(accepted)}",
    )


def grade_citations(case: EvalCase, answer: Answer, text_dir: Path) -> Grade:
    """Every expected (document, section) is cited, and every cited line range exists
    in `text_dir/<document>.txt`.

    Every citation must also pass `citation_problem`: quote in the lines, section the
    heading above them or named in them. A cited section matches an expected one if it
    is the same or more specific: "7.06(a)" is met by "7.06(a)(i)", not by "7.06" or
    "7.06(b)". Citations matching no expected one (extra or superseded) don't fail
    the grade; they are listed in `warning`.

    Proves: the locators are real and point where expected. Does not prove: that the
    cited lines support the answer, or that a cited subsection like "(a)" is the one
    the lines are in (`citation_problem` checks section numbers only).
    """
    problems = [
        f"{_label(c)}: {problem}"
        for c in answer.citations
        if (problem := citation_problem(c, text_dir))
    ]
    if problems:
        return Grade(False, "; ".join(problems))
    missing = [
        f"{e.document} §{e.section}"
        for e in case.expected_citations
        if not any(
            c.document == e.document and _covers(e.section, c.section)
            for c in answer.citations
        )
    ]
    extra = [
        _label(c)
        for c in answer.citations
        if not any(
            c.document == e.document and _covers(e.section, c.section)
            for e in case.expected_citations
        )
    ]
    warning = f"extra: {', '.join(extra)}" if extra else ""
    if missing:
        reason = f"missing: {', '.join(missing)}"
        return Grade(False, f"{reason}; {warning}" if warning else reason)
    return Grade(True, "all expected citations present", warning)


def normalize_value(value: str) -> str:
    """Lowercase, ratios as "a:b" with plain numbers, single spaces, no trailing
    punctuation: "3.75 to 1.00", "3.75:1" and "3.75x" all become "3.75:1"."""
    value = _RATIO.sub(
        lambda m: f"{_number(m.group(1))}:{_number(m.group(2) or '1')}",
        value.lower(),
    )
    return " ".join(value.split()).rstrip(".,;:!? ")


def _number(text: str) -> str:
    """'2.00' -> '2', '3.750' -> '3.75'."""
    return text.rstrip("0").rstrip(".") if "." in text else text


def _section(text: str) -> str | None:
    match = _SECTION.search(re.sub(r"\s+", "", text.lower()))
    return match.group() if match else None


def _covers(expected: str, cited: str) -> bool:
    """`cited` is `expected` or a subsection of it."""
    e, c = _section(expected), _section(cited)
    return e is not None and c is not None and (c == e or c.startswith(e + "("))


def _label(citation: Citation) -> str:
    start, end = citation.lines
    return f"{citation.document} §{citation.section} [{start}-{end}]"
