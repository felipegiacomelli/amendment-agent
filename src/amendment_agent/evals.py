"""The eval runner: run each case through the agent, grade its answer file, report.

Reports, never gates. Results are comparable only on the same case set and grader
version, so every results file records both.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import attrs
from anthropic import Anthropic

from amendment_agent.agent import run_agent
from amendment_agent.graders import Grade, grade_answer, grade_citations
from amendment_agent.recorder import RunRecorder, git_revision, new_id
from amendment_agent.schema import ANSWER_FILENAME, EvalCase, load_answer

COMPARABILITY = (
    "Compare results only when cases_sha256 and the git commit (graders) match; "
    "otherwise rerun the older version on the current case set."
)


@attrs.frozen
class CaseResult:
    case_id: str
    category: str
    answer: Grade
    citations: Grade
    stop_reason: str
    turns: int
    cost_usd: float
    cost_complete: bool
    run_id: str


def _fail_closed(grade: Callable[[], Grade]) -> Grade:
    """A grader that raises is a failed grade, never a pass."""
    try:
        return grade()
    except NotImplementedError:
        raise
    except Exception as error:
        return Grade(
            passed=False, reason=f"grader raised {type(error).__name__}: {error}"
        )


def run_case(
    case: EvalCase,
    *,
    client: Anthropic,
    model: str,
    max_turns: int,
    max_cost_usd: float,
    config: dict[str, Any],
    runs_dir: Path,
    outputs_dir: Path,
    text_dir: Path,
) -> CaseResult:
    recorder = RunRecorder(
        runs_dir, model, case.question, label=case.id, text_dir=text_dir, extra=config
    )
    output_dir = outputs_dir / recorder.run_id

    def row(answer: Grade, citations: Grade, stop_reason: str) -> CaseResult:
        return CaseResult(
            case.id,
            str(case.category),
            answer,
            citations,
            stop_reason,
            recorder.turns,
            recorder.total_cost_usd,
            recorder.cost_complete,
            recorder.run_id,
        )

    try:
        with recorder:  # an escaping exception is recorded as run_end/error
            stop_reason = run_agent(
                case.question,
                client=client,
                model=model,
                recorder=recorder,
                output_dir=output_dir,
                max_turns=max_turns,
                max_cost_usd=max_cost_usd,
            )
    except NotImplementedError:
        raise
    except Exception as error:
        failed = Grade(
            passed=False, reason=f"agent raised {type(error).__name__}: {error}"
        )
        return row(failed, failed, "error")
    try:
        answer = load_answer(output_dir / ANSWER_FILENAME)
    except ValueError as error:
        failed = Grade(passed=False, reason=str(error))
        return row(failed, failed, str(stop_reason))
    return row(
        _fail_closed(lambda: grade_answer(case, answer)),
        _fail_closed(lambda: grade_citations(case, answer, text_dir)),
        str(stop_reason),
    )


def format_table(results: list[CaseResult]) -> str:
    def mark(grade: Grade) -> str:
        return "✓" if grade.passed else "✗"

    lines = [
        f"{'case':<26} {'category':<16} {'answer':<6} {'cite':<5} {'stop':<15} {'turns':>5} {'cost $':>9}"
    ]
    for r in results:
        cost = f"{r.cost_usd:.4f}{'' if r.cost_complete else '?'}"
        lines.append(
            f"{r.case_id:<26} {r.category:<16} {mark(r.answer):<6} {mark(r.citations):<5} "
            f"{r.stop_reason:<15} {r.turns:>5} {cost:>9}"
        )
    n = len(results)
    total = sum(r.cost_usd for r in results)
    lines.append(
        f"answer {sum(r.answer.passed for r in results)}/{n}  "
        f"citations {sum(r.citations.passed for r in results)}/{n}  "
        f"total ${total:.4f}  avg ${total / n if n else 0:.4f}"
        + (
            "  (? = cost incomplete)"
            if any(not r.cost_complete for r in results)
            else ""
        )
    )
    for r in results:
        for kind, grade in (("answer", r.answer), ("citations", r.citations)):
            if not grade.passed:
                lines.append(f"  {r.case_id} {kind}: {grade.reason}")
    return "\n".join(lines)


def write_results(
    results: list[CaseResult],
    *,
    eval_id: str,
    model: str,
    cases_path: Path,
    cases_sha256: str,
    config: dict[str, Any],
    results_dir: Path,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{eval_id}.json"
    payload = {
        "eval_id": eval_id,
        "model": model,
        "git": git_revision(),
        "cases_path": str(cases_path),
        "cases_sha256": cases_sha256,
        "case_ids": [r.case_id for r in results],
        "config": config,
        "comparability": COMPARABILITY,
        "results": [attrs.asdict(r) for r in results],
    }
    with path.open("x", encoding="utf-8") as file:  # never overwrite earlier results
        file.write(json.dumps(payload, indent=2, default=str) + "\n")
    return path


def run_eval(
    cases: list[EvalCase],
    *,
    client: Anthropic,
    model: str,
    max_turns: int,
    max_cost_usd: float,
    config: dict[str, Any],
    cases_path: Path,
    runs_dir: Path,
    outputs_dir: Path,
    results_dir: Path,
    text_dir: Path,
) -> list[CaseResult]:
    """Run every case, print the table, write a results file. Reports, never gates."""
    eval_id = new_id(model)
    cases_sha256 = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    config = config | {"eval_id": eval_id, "cases_sha256": cases_sha256}
    results = [
        run_case(
            case,
            client=client,
            model=model,
            max_turns=max_turns,
            max_cost_usd=max_cost_usd,
            config=config,
            runs_dir=runs_dir,
            outputs_dir=outputs_dir,
            text_dir=text_dir,
        )
        for case in cases
    ]
    print(format_table(results))
    path = write_results(
        results,
        eval_id=eval_id,
        model=model,
        cases_path=cases_path,
        cases_sha256=cases_sha256,
        config=config,
        results_dir=results_dir,
    )
    print(f"results: {path}")
    print(COMPARABILITY)
    return results
