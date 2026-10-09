"""Command-line entry point: amendment-agent fetch | extract | ask | eval."""

import argparse
import sys
from pathlib import Path

from anthropic import Anthropic

from amendment_agent.agent import StopReason, run_agent
from amendment_agent.edgar import FetchError, fetch_all
from amendment_agent.extract import extract_all
from amendment_agent.manifest import load_manifest
from amendment_agent.recorder import PRICES, RunRecorder
from amendment_agent.schema import ANSWER_FILENAME, load_answer
from amendment_agent.settings import (
    MANIFEST_PATH,
    OUTPUT_DIR,
    RAW_DIR,
    RUNS_DIR,
    TEXT_DIR,
    Settings,
)


def fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def cmd_fetch(settings: Settings) -> int:
    user_agent = (settings.sec_user_agent or "").strip()
    if not user_agent:
        return fail(
            "SEC_USER_AGENT is not set; add 'Name email' to .env (see .env.example)"
        )
    try:
        fetch_all(load_manifest(MANIFEST_PATH), RAW_DIR, user_agent)
    except FetchError as error:
        return fail(str(error))
    return 0


def cmd_extract() -> int:
    try:
        extract_all(load_manifest(MANIFEST_PATH), RAW_DIR, TEXT_DIR)
    except FileNotFoundError as error:
        return fail(str(error))
    return 0


def prepare(settings: Settings, model_flag: str | None) -> tuple[Anthropic, str] | str:
    """The client and model for ask/eval, or an error message."""
    model = model_flag or settings.agent_model
    if model not in PRICES:
        return f"unknown model {model!r}; supported: {', '.join(sorted(PRICES))}"
    key = (
        settings.anthropic_api_key.get_secret_value().strip()
        if settings.anthropic_api_key
        else ""
    )
    if not key:
        return "ANTHROPIC_API_KEY is not set; add it to .env (see .env.example)"
    client = Anthropic(
        api_key=key,
        timeout=settings.agent_request_timeout_s,
        max_retries=settings.agent_max_retries,
    )
    return client, model


def run_config(settings: Settings) -> dict[str, object]:
    """Limits and client settings recorded in every run_start. No secrets."""
    return {
        "max_turns": settings.agent_max_turns,
        "max_cost_usd": settings.agent_max_cost_usd,
        "request_timeout_s": settings.agent_request_timeout_s,
        "max_retries": settings.agent_max_retries,
    }


def print_answer(output_dir: Path) -> None:
    try:
        answer = load_answer(output_dir / ANSWER_FILENAME)
    except ValueError as error:
        print(f"answer: none ({error})")
        return
    print(f"answer: [{answer.status}] {answer.value or ''}".rstrip())
    print(f"        {answer.explanation}")
    for citation in answer.citations:
        start, end = citation.lines
        print(f"  cite: {citation.document} §{citation.section}, lines {start}-{end}")


def cmd_ask(settings: Settings, question: str, model_flag: str | None) -> int:
    prepared = prepare(settings, model_flag)
    if isinstance(prepared, str):
        return fail(prepared)
    client, model = prepared
    with RunRecorder(
        RUNS_DIR, model, question, label="ask", extra=run_config(settings)
    ) as recorder:
        output_dir = OUTPUT_DIR / recorder.run_id
        stop_reason = run_agent(
            question,
            client=client,
            model=model,
            recorder=recorder,
            output_dir=output_dir,
            max_turns=settings.agent_max_turns,
            max_cost_usd=settings.agent_max_cost_usd,
        )
    incomplete = (
        ""
        if recorder.cost_complete
        else " (incomplete: a request has no recorded response)"
    )
    print(f"stop:   {stop_reason}")
    print(f"run:    {recorder.path}")
    print(
        f"turns:  {recorder.turns}   cost: ${recorder.total_cost_usd:.4f}{incomplete}"
    )
    print_answer(output_dir)
    return 0 if stop_reason is StopReason.ANSWERED else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amendment-agent",
        description="Answer questions about a credit agreement family, citing the provision that applies.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "fetch", help="download the manifest's filings from SEC EDGAR into data/raw/"
    )
    commands.add_parser(
        "extract",
        help="convert data/raw/ into line-oriented text and _index.md in data/text/",
    )
    ask = commands.add_parser("ask", help="answer one question with the agent")
    ask.add_argument("question")
    ask.add_argument("--model", help="override AGENT_MODEL")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    match args.command:
        case "fetch":
            return cmd_fetch(settings)
        case "extract":
            return cmd_extract()
        case "ask":
            return cmd_ask(settings, args.question, args.model)
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
