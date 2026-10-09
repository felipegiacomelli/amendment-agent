"""Command-line entry point: amendment-agent fetch | extract | ask | eval."""

import argparse
import sys

from amendment_agent.edgar import FetchError, fetch_all
from amendment_agent.extract import extract_all
from amendment_agent.manifest import load_manifest
from amendment_agent.settings import MANIFEST_PATH, RAW_DIR, TEXT_DIR, Settings


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    match args.command:
        case "fetch":
            return cmd_fetch(settings)
        case "extract":
            return cmd_extract()
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
