"""Deterministic eval graders. USER-OWNED: the scaffold defines the contract only.

The eval runner calls these only with a valid `Answer`; a missing or malformed
answer file has already failed. Graders must still fail closed: any doubt is a fail.
"""

from pathlib import Path

import attrs

from amendment_agent.schema import Answer, EvalCase


@attrs.frozen
class Grade:
    passed: bool
    reason: str


def grade_answer(case: EvalCase, answer: Answer) -> Grade:
    """Status matches `case.expected_status` and, if answered, the value is accepted.

    Proves: the agent's stated value equals one of `case.accepted_values` under your
    normalisation. Does not prove: that the explanation is right.
    """
    raise NotImplementedError


def grade_citations(case: EvalCase, answer: Answer, text_dir: Path) -> Grade:
    """Every expected (document, section) is cited, and every cited line range exists
    in `text_dir/<document>.txt`.

    Proves: the locators are real and point where expected. Does not prove: that the
    cited lines support the answer.
    """
    raise NotImplementedError
