"""Reading the corpus text and checking citations against it.

`write_answer` uses `citation_problem` to reject an answer, and `grade_citations` to
grade one, so this file is part of the grading fingerprint (`evals.GRADING_FILES`):
changing a check here makes earlier results incomparable. Tool descriptions stay in
`tools.py`, so editing them doesn't.
"""

import re
from pathlib import Path

from amendment_agent.schema import Citation

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


def resolve(path: str, root: Path) -> Path | None:
    """`root/path` if it stays inside `root` (symlinks and `..` resolved), else None."""
    base = root.resolve()
    target = (base / path).resolve()
    return target if target.is_relative_to(base) else None


def read_lines(file: Path) -> list[str]:
    """The one place a file is split into lines."""
    return file.read_text(encoding="utf-8").splitlines()


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
    and the section is either the heading the lines sit under or a section the lines
    name ("Section 7.06(a) ... is amended", in an amendment). Sections compare by
    number: a "(a)" subsection isn't checked. Lines are split as read_file splits
    them. Does not check that the lines support the answer.
    """
    # TODO: two more deterministic checks. (1) The agent read the cited lines this
    # run: read_file records what it returned in a per-run read log that run_agent
    # creates and passes through execute_tool. (2) The answer's value appears in a
    # quote, using the value normalisation grade_answer needs. Whether the lines
    # support the answer needs an LLM judge, so it stays in the eval.
    name = citation.document
    file = resolve(f"{name}.txt", text_dir)
    if file is None or not file.is_file():
        return f"no document {name!r} (use the file name without .txt)"
    lines = read_lines(file)
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
    if cited is None:
        return f"section {citation.section!r} has no section number, e.g. 6.12"
    # "Section 7.06" but not "Section 7.061" or, for "7", "Section 7.06".
    named = re.compile(rf"\bSections? {re.escape(cited.group())}(?![.\d]\d)")
    if cited.group() != actual and not named.search(" ".join(lines[start - 1 : end])):
        return (
            f"{name} line {start} is in section {actual}, and lines {start}-{end} "
            f"don't mention Section {cited.group()}"
        )
    if (last := section_at(lines, end)) != actual:
        return (
            f"{name} lines {start}-{end} run from section {actual} into {last}; "
            "cite one section"
        )
    return None
