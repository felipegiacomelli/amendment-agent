"""The agent's tools over a folder of files. USER-OWNED.

Reads stay inside data/text/; the only write is the run's answer.json. A path
outside its root returns an error result the model can see; tools never raise.
Line numbers are 1-based everywhere, and every tool splits a file into lines the
same way (`_read_lines`), so a grep hit's line N is read_file's line N.
"""

import re
from collections.abc import Callable
from pathlib import Path

import attrs
from anthropic.types import ToolParam
from pydantic import ValidationError

from amendment_agent.schema import ANSWER_FILENAME, Answer, AnswerStatus, Citation
from amendment_agent.settings import OUTPUT_DIR, TEXT_DIR

# ponytail: fixed caps keep one call from flooding the context; tune if runs need more.
MAX_GREP_HITS = 100
MAX_HIT_CHARS = 400
MAX_READ_LINES = 200
# Shorter quotes ("the Company") match almost anywhere and prove nothing.
MIN_QUOTE_CHARS = 20
# Lines are whole blocks (a paragraph, a definition), so a provision fits in a few.
# A wide range would let any true quote from far away pass the quote check.
MAX_CITATION_LINES = 20

# Text files are one block per line, so a section heading is a line that starts with
# a section number and a capitalised word: "7.11 Burdensome Agreements. ..." in the
# credit agreement (original, and the restated one in amendment-1), "1.Amendments to
# Credit Agreement. ..." in the amendments. Not table-of-contents rows ("7.11 |
# Burdensome Agreements 101") or wrapped text ("9.10 of the Credit Agreement").
# ponytail: numbered items in the exhibit forms also match; citing an exhibit form
# reports its item number as the section. Parse exhibits if that ever matters.
_HEADING = re.compile(r"^(?P<number>\d{1,2}\.\d{2}(?= [A-Z])|\d{1,2}(?=\.[A-Z]))")
# The section number a citation names: "7.06" in "Section 7.06(a)", "1" in "1(b)".
_SECTION_NUMBER = re.compile(r"\d{1,2}(?:\.\d{2})?")
_PUNCTUATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
)


@attrs.frozen
class ToolResult:
    text: str
    is_error: bool


# Names, descriptions and input schemas are prompt engineering: user-owned.
# Every "" is a description to write. Property names must match the function
# arguments (a test checks this). `strict` makes the API guarantee inputs match the
# schema; it rejects `minimum`, `minLength` and array-length keywords, so those
# rules are checked in code.
TOOL_SCHEMAS: list[ToolParam] = [
    {
        "name": "list_files",
        "description": "",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": ""},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "grep",
        "description": "",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": ""},
                "path": {"type": "string", "description": ""},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": "",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": ""},
                "start_line": {"type": "integer", "description": ""},
                "end_line": {"type": "integer", "description": ""},
            },
            "required": ["path", "start_line", "end_line"],
            "additionalProperties": False,
        },
    },
    {
        # Mirrors schema.Answer. Rules JSON Schema can't express here (an answered
        # status needs a value and a citation, lines has two items) are checked by
        # Answer in write_answer and come back as error results.
        "name": "write_answer",
        "description": "",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": [s.value for s in AnswerStatus],
                    "description": "",
                },
                "value": {"type": "string", "description": ""},
                "explanation": {"type": "string", "description": ""},
                "citations": {
                    "type": "array",
                    "description": "",
                    "items": {
                        "type": "object",
                        "properties": {
                            "document": {"type": "string", "description": ""},
                            "section": {"type": "string", "description": ""},
                            "lines": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "",
                            },
                            "quote": {"type": "string", "description": ""},
                        },
                        "required": ["document", "section", "lines", "quote"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["status", "explanation", "citations"],
            "additionalProperties": False,
        },
    },
]


def _error(message: str) -> ToolResult:
    return ToolResult(text=f"error: {message}", is_error=True)


def _resolve(path: str, root: Path) -> Path | None:
    """`root/path` if it stays inside `root` (symlinks and `..` resolved), else None."""
    base = root.resolve()
    target = (base / path).resolve()
    return target if target.is_relative_to(base) else None


def _read_lines(file: Path) -> list[str]:
    """The one place a file is split into lines."""
    return file.read_text(encoding="utf-8").splitlines()


def _files_under(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*") if p.is_file())


def list_files(path: str = "", *, root: Path = TEXT_DIR) -> ToolResult:
    """List the documents under `root/path`."""
    target = _resolve(path, root)
    if target is None or not target.is_dir():
        return _error(f"not a directory under the document root: {path!r}")
    base = root.resolve()
    names = [str(p.relative_to(base)) for p in _files_under(target)]
    return ToolResult(text="\n".join(names) or "(no files)", is_error=False)


def grep(pattern: str, path: str = "", *, root: Path = TEXT_DIR) -> ToolResult:
    """Search files under `root/path`; report matches with 1-based line numbers."""
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return _error(f"invalid regex {pattern!r}: {e}")
    target = _resolve(path, root)
    if target is None or not target.exists():
        return _error(f"no such file or directory under the document root: {path!r}")
    base = root.resolve()
    hits: list[str] = []
    for file in _files_under(target):
        for lineno, line in enumerate(_read_lines(file), start=1):
            if not regex.search(line):
                continue
            if len(hits) == MAX_GREP_HITS:
                hits.append(
                    f"(stopped at {MAX_GREP_HITS} hits; narrow the pattern or path)"
                )
                return ToolResult(text="\n".join(hits), is_error=False)
            if len(line) > MAX_HIT_CHARS:
                line = line[:MAX_HIT_CHARS] + "..."
            hits.append(f"{file.relative_to(base)}:{lineno}: {line}")
    return ToolResult(text="\n".join(hits) or "(no matches)", is_error=False)


def read_file(
    path: str, start_line: int, end_line: int, *, root: Path = TEXT_DIR
) -> ToolResult:
    """Return lines `start_line..end_line` (1-based, inclusive) of `root/path`."""
    target = _resolve(path, root)
    if target is None or not target.is_file():
        return _error(f"no such file under the document root: {path!r}")
    lines = _read_lines(target)
    if start_line < 1 or end_line < start_line or start_line > len(lines):
        return _error(
            f"invalid range {start_line}-{end_line}; {path} has lines 1-{len(lines)}"
        )
    end = min(end_line, len(lines), start_line + MAX_READ_LINES - 1)
    body = "\n".join(
        f"{n}: {line}"
        for n, line in enumerate(lines[start_line - 1 : end], start=start_line)
    )
    header = f"{path} lines {start_line}-{end} of {len(lines)}"
    if end < min(end_line, len(lines)):
        header += f" (capped at {MAX_READ_LINES} lines per call)"
    return ToolResult(text=f"{header}\n{body}", is_error=False)


def _normalize(text: str) -> str:
    """Straight quotes and dashes, single spaces: what a copied quote may lose."""
    return " ".join(text.translate(_PUNCTUATION).split())


def section_at(lines: list[str], line: int) -> str | None:
    """Number of the section heading at or before 1-based `line`, if any."""
    for text in reversed(lines[:line]):
        if " | " not in text and (match := _HEADING.match(text)):
            return match["number"]
    return None


def citation_problem(citation: Citation, text_dir: Path) -> str | None:
    """Why `citation` isn't backed by `text_dir`, or None if it is.

    Checks that the document exists, the range is inside it, at most
    MAX_CITATION_LINES long and within one section, the quote appears in those lines,
    and the section matches that section's heading (by number: a "(a)" subsection
    isn't checked). Lines are split as read_file splits them. Does not check that the
    lines support the answer.
    """
    # TODO: two more deterministic checks. (1) The agent read the cited lines this
    # run: read_file records what it returned in a per-run read log that run_agent
    # creates and passes through execute_tool. (2) The answer's value appears in a
    # quote, using the value normalisation grade_answer needs. Whether the lines
    # support the answer needs an LLM judge, so it stays in the eval.
    name = citation.document
    file = _resolve(f"{name}.txt", text_dir)
    if file is None or not file.is_file():
        return f"no document {name!r} (use the file name without .txt)"
    lines = _read_lines(file)
    start, end = citation.lines
    if end > len(lines):
        return f"{name} lines {start}-{end} are past its end ({len(lines)} lines)"
    if end - start + 1 > MAX_CITATION_LINES:
        return (
            f"{name} lines {start}-{end} span {end - start + 1} lines; cite at most "
            f"{MAX_CITATION_LINES}, around the quoted text"
        )
    quote = _normalize(citation.quote)
    if len(quote) < MIN_QUOTE_CHARS:
        return f"quote for {name} lines {start}-{end} is under {MIN_QUOTE_CHARS} characters"
    if quote not in _normalize(" ".join(lines[start - 1 : end])):
        return f"quote not found in {name} lines {start}-{end}; copy it exactly"
    cited = _SECTION_NUMBER.search(citation.section)
    actual = section_at(lines, start)
    if actual is None:
        return f"{name} line {start} is not under a numbered section heading"
    if cited is None or cited.group() != actual:
        return f"{name} line {start} is in section {actual}, not {citation.section!r}"
    if (last := section_at(lines, end)) != actual:
        return (
            f"{name} lines {start}-{end} run from section {actual} into {last}; "
            "cite one section"
        )
    return None


def write_answer(
    status: str,
    explanation: str,
    citations: list[dict[str, object]],
    value: str | None = None,
    *,
    root: Path = OUTPUT_DIR,
    text_dir: Path = TEXT_DIR,
) -> ToolResult:
    """Validate the answer and write `root/answer.json`.

    Valid means it passes `schema.Answer` and every citation points at real lines in
    `text_dir`. The loop passes the run's output_dir as root. A second call
    overwrites the first.
    """
    try:
        answer = Answer.model_validate(
            {
                "status": status,
                "value": value,
                "explanation": explanation,
                "citations": citations,
            }
        )
    except ValidationError as e:
        return _error(f"answer not written, fix and call again: {e}")
    problems = [
        problem
        for citation in answer.citations
        if (problem := citation_problem(citation, text_dir))
    ]
    if problems:
        return _error("answer not written, fix and call again: " + "; ".join(problems))
    root.mkdir(parents=True, exist_ok=True)
    (root / ANSWER_FILENAME).write_text(
        answer.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return ToolResult(text=f"wrote {ANSWER_FILENAME}", is_error=False)


TOOL_MAP: dict[str, Callable[..., ToolResult]] = {
    "list_files": list_files,
    "grep": grep,
    "read_file": read_file,
    "write_answer": write_answer,
}


def execute_tool(
    name: str, tool_input: dict[str, object], *, output_dir: Path
) -> ToolResult:
    """Run the named tool. Unknown names and invalid input are error results."""
    fn = TOOL_MAP.get(name)
    if fn is None:
        return _error(f"unknown tool {name!r}; available: {', '.join(TOOL_MAP)}")
    reserved = sorted(tool_input.keys() & {"root", "text_dir"})
    if reserved:  # directories are ours to set, never the model's
        return _error(f"unexpected argument {reserved[0]!r}")
    kwargs = dict(tool_input)
    if name == "write_answer":
        kwargs["root"] = output_dir
    try:
        return fn(**kwargs)
    except (TypeError, ValueError, OSError) as e:  # bad args or types, unreadable file
        return _error(f"{name} failed: {e}")
