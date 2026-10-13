"""The agent's tools over a folder of files. USER-OWNED.

Reads stay inside data/text/; the only write is the run's answer.json. A path
outside its root returns an error result the model can see; tools never raise.
Line numbers are 1-based everywhere, and every tool splits a file into lines the
same way (`read_lines`), so a grep hit's line N is read_file's line N.
"""

import re
from collections.abc import Callable
from pathlib import Path

import attrs
from anthropic.types import ToolParam
from pydantic import ValidationError

from amendment_agent.citations import citation_problem, read_lines, resolve
from amendment_agent.schema import ANSWER_FILENAME, Answer, AnswerStatus
from amendment_agent.settings import OUTPUT_DIR, TEXT_DIR

# ponytail: fixed caps keep one call from flooding the context; tune if runs need more.
MAX_GREP_HITS = 100
MAX_HIT_CHARS = 400
MAX_READ_LINES = 200


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
        "description": (
            "List the documents available to search: the credit agreement, its "
            "amendments, and _index.md, which gives each document's title, its "
            "'dated as of' date (this sets the documents' order, not the filing "
            "date), what it amends, and its line count. Paths are relative to the "
            "document folder, one per line. Use it at the start to see what exists; "
            "read _index.md with read_file before deciding which version applies."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Subfolder to list, relative to the document folder. Empty "
                        "lists everything."
                    ),
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "grep",
        "description": (
            "Search the documents for a Python regular expression, matched "
            "case-insensitively against each line. Every line is a whole block of "
            "the filing (a paragraph, a definition, a table row), so a hit locates "
            "a provision. Returns one hit per line as 'file:line: text', with line "
            "numbers 1-based and identical to read_file's; text over 400 characters "
            "is cut with '...'. Stops after 100 hits and says so: narrow the pattern "
            "or the path then. Returns '(no matches)' when nothing matches. Use it "
            "to find where a term, defined word or section number appears, then "
            "read_file around the hit to see the full provision."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Python regex, case-insensitive. Escape literal dots and "
                        "parentheses, e.g. 'Section 6\\.12' or 'Fixed Charge'."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": (
                        "A file (e.g. 'amendment-3.txt') or folder to search, "
                        "relative to the document folder. Empty searches every "
                        "document."
                    ),
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a range of lines from one document. Returns a header "
            "'<path> lines <start>-<end> of <total>', then each line prefixed with "
            "its 1-based number; those numbers are what citations use. Returns at "
            "most 200 lines per call and says when it capped the range; read the "
            "rest with another call. Lines '[Page N]' mark page breaks in the "
            "filing, and table cells are joined by ' | '. A range starting before "
            "line 1 or past the end of the file is an error that states the file's "
            "line count."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "File to read, as list_files shows it, e.g. 'original.txt'."
                    ),
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to return, 1-based.",
                },
                "end_line": {
                    "type": "integer",
                    "description": (
                        "Last line to return, inclusive. A value past the end of "
                        "the file stops at the last line."
                    ),
                },
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
        "description": (
            "Record the final answer to the question; this is how the run delivers "
            "it, so finish every run with this call. Use status 'answered' with a "
            "value and at least one citation when the documents settle the "
            "question, or 'insufficient_evidence' with no value when they don't. "
            "Each citation is checked against the documents before anything is "
            "written: the document must exist, the lines must be at most 20 and "
            "inside one section, the quote must appear in those lines, and the "
            "section must be the heading those lines sit under. If any check fails "
            "the answer is not written and the error says which citation is wrong; "
            "fix it and call again. A later call replaces an earlier answer."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": [s.value for s in AnswerStatus],
                    "description": (
                        "'answered' if the documents give the answer for the "
                        "period asked about; 'insufficient_evidence' if they "
                        "don't, rather than a guess."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": (
                        "Required when answered, omitted otherwise. The short "
                        "answer itself, as the documents state it, e.g. '2.50:1', "
                        "'$75,000,000' or '2015-06-01'; no explanation here."
                    ),
                },
                "explanation": {
                    "type": "string",
                    "description": (
                        "A few sentences on why this is the provision in force for "
                        "the period asked about: which document and amendment "
                        "apply, and why later or earlier ones don't. For "
                        "insufficient_evidence, what was searched and what is "
                        "missing."
                    ),
                },
                "citations": {
                    "type": "array",
                    "description": (
                        "The evidence for the value: the provision in force, plus "
                        "any amendment that put it in force. Empty for "
                        "insufficient_evidence."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "document": {
                                "type": "string",
                                "description": (
                                    "Document id: the file name without .txt, "
                                    "e.g. 'original' or 'amendment-1'."
                                ),
                            },
                            "section": {
                                "type": "string",
                                "description": (
                                    "The credit agreement section the citation is "
                                    "about, e.g. '6.12(b)'. In the agreement "
                                    "itself, the heading the cited lines sit "
                                    "under. In an amendment, the section the "
                                    "cited lines amend, which those lines must "
                                    'name ("Section 6.12(b) of the Credit '
                                    'Agreement is hereby amended..."); for '
                                    "amendment text that changes no section, the "
                                    "amendment's own heading number, e.g. '2'."
                                ),
                            },
                            "lines": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": (
                                    "[start, end]: 1-based inclusive line range "
                                    "from read_file, at most 20 lines, all in the "
                                    "same section."
                                ),
                            },
                            "quote": {
                                "type": "string",
                                "description": (
                                    "Text copied exactly from the cited lines, at "
                                    "least 20 characters: the words that state the "
                                    "value or make the change. Straight or curly "
                                    "quotes and spacing may differ; nothing else."
                                ),
                            },
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


def _files_under(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*") if p.is_file())


def list_files(path: str = "", *, root: Path = TEXT_DIR) -> ToolResult:
    """List the documents under `root/path`."""
    target = resolve(path, root)
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
    target = resolve(path, root)
    if target is None or not target.exists():
        return _error(f"no such file or directory under the document root: {path!r}")
    base = root.resolve()
    hits: list[str] = []
    for file in _files_under(target):
        for lineno, line in enumerate(read_lines(file), start=1):
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
    target = resolve(path, root)
    if target is None or not target.is_file():
        return _error(f"no such file under the document root: {path!r}")
    lines = read_lines(target)
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
    name: str, tool_input: dict[str, object], *, output_dir: Path, text_dir: Path
) -> ToolResult:
    """Run the named tool over `text_dir` (writes go to `output_dir`). Unknown names
    and invalid input are error results."""
    fn = TOOL_MAP.get(name)
    if fn is None:
        return _error(f"unknown tool {name!r}; available: {', '.join(TOOL_MAP)}")
    reserved = sorted(tool_input.keys() & {"root", "text_dir"})
    if reserved:  # directories are ours to set, never the model's
        return _error(f"unexpected argument {reserved[0]!r}")
    kwargs = dict(tool_input)
    if name == "write_answer":
        kwargs |= {"root": output_dir, "text_dir": text_dir}
    else:
        kwargs["root"] = text_dir
    try:
        return fn(**kwargs)
    except (TypeError, ValueError, OSError) as e:  # bad args or types, unreadable file
        return _error(f"{name} failed: {e}")
