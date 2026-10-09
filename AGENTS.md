# AGENTS.md

Instructions for coding agents working in this repo. There is deliberately no
`CLAUDE.md`. Don't create one.

## Project

An autonomous agent that answers questions about one credit agreement family (an
original agreement plus its amendments, from SEC EDGAR). For each question it finds
the provision that applies to the period asked about, within this document set, and
cites document, section and line range. Raw Anthropic Messages API with a
hand-written tool loop. No agent frameworks.

## The core is hand-written: don't implement it

These are written by the maintainer, by hand, as the point of the project:

- `src/amendment_agent/agent.py`: `run_agent`, `SYSTEM_PROMPT`, `StopReason`
- `src/amendment_agent/tools.py`: the four tools, `execute_tool`, `TOOL_SCHEMAS`
- `src/amendment_agent/graders.py`: `grade_answer`, `grade_citations`
- their tests, and `evals/cases.jsonl` beyond the first example

You may explain, review and suggest changes to them. Don't write or change their
bodies unless the maintainer explicitly asks in the current session. Plumbing
(fetcher, extraction, recorder, schema, eval runner, CLI, tooling) is fair game.

## Commands

| Command | What it does |
|---|---|
| `make install` | `uv sync` + pre-commit hooks |
| `make lint` | ruff, black `--check`, `mypy --strict` |
| `make fmt` | ruff `--fix` + black |
| `make test` | pytest with the coverage floor |
| `make check` | lint + test |
| `uv run amendment-agent fetch` | download the filings in `data/manifest.yaml` to `data/raw/` |
| `uv run amendment-agent extract` | `data/raw/` → line-oriented text and `_index.md` in `data/text/` |
| `uv run amendment-agent ask "<q>" [--model M]` | one run → `runs/<run_id>.jsonl`, `outputs/<run_id>/answer.json` |
| `uv run amendment-agent eval [--model M] [--case ID]` | all cases → table + `evals/results/<eval_id>.json` |

## Conventions

- Python 3.12 idioms only:
  - no `from __future__ import annotations`;
  - no `typing.Optional` or `typing.List`;
  - PEP 695 `type` aliases, `typing.override`, `pathlib`;
  - break import cycles by moving types into their own module.
- uv, with versions pinned in `uv.lock`. CI uses `uv sync --frozen`.
- black (88), ruff (`E F I B UP`; ruff `I` replaces isort), `mypy --strict` on `src`
  and `tests`.
- `attrs` (`@attrs.frozen`) for plain data classes, never `dataclasses`. pydantic only
  at trust boundaries: settings, manifest, `schema.py`.
- Tests are offline and deterministic: no network, no API key, never the real
  `run_agent`. The coverage floor only ever goes up.
- Contracts:
  - text format: one block per line, `[Page N]` markers, table cells joined by ` | `;
  - the agent's answer is `outputs/<run_id>/answer.json`, matching `schema.Answer`,
    and the file is authoritative;
  - document order comes from the manifest's `dated`, not `filing_date`.
- Only models in `recorder.PRICES` are accepted, so cost is never computed from
  incomplete prices.

## Evals

- Compare results only when `cases_sha256` and the git commit (graders) match. If the
  case set changed, rerun the older version on the new set or report them separately.
- Regression cases found in real runs may be added. Add them openly, noting the run
  that revealed them. Keep earlier results files.
- Graders fail closed. The eval reports and never gates CI; CI has no API key.

## Commits

- One logical change per commit, with its tests. Conventional messages:
  `type(scope): imperative description`.
- No `Co-Authored-By` or other AI attribution in commit messages.
- Never commit `.env`, `data/raw/`, `data/text/`, or `runs/` and `outputs/` contents,
  unless the maintainer force-adds a run as evidence.
