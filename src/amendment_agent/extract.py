"""Turn raw EDGAR exhibits into line-oriented text for the agent's tools.

Format: one block (paragraph, heading, table row) per line, so a line number is a
stable citation anchor. Line 1 is `[Page 1]`; each page break adds `[Page N]`,
counting pages as the filing renders them (not the printed footer numbers, which stay
in the text as their own lines). Table rows are their non-empty cells joined by " | ".
# ponytail: no structure-aware parsing or printed-page mapping; that is the user's
# rung-4 work.
"""

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import override

from amendment_agent.edgar import raw_path
from amendment_agent.manifest import Manifest

BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "blockquote",
        "br",
        "center",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "hr",
        "li",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "tfoot",
        "thead",
        "ul",
    }
)
CELL_TAGS = frozenset({"td", "th"})
SKIP_TAGS = frozenset({"head", "script", "style"})
VOID_TAGS = frozenset({"br", "hr", "img", "input", "meta", "link", "col", "wbr"})
PAGE_BREAK = re.compile(r"page-break-(before|after)\s*:\s*always", re.IGNORECASE)
PAGE_MARKER = re.compile(r"\[Page \d+\]")
PAGE_TAG = re.compile(r"^\s*<PAGE>\s*\d*\s*$", re.IGNORECASE)
INDEX_NAME = "_index.md"


def clean(text: str) -> str:
    """Collapse all whitespace (NBSP included) and drop zero-width spaces."""
    return " ".join(text.replace("\u200b", "").split())


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = ["[Page 1]"]
        self.page = 1
        self._buffer: list[str] = []
        self._cells: list[list[str]] | None = None  # set while inside a <tr>
        self._skip_depth = 0
        self._open: list[tuple[str, bool]] = []  # (tag, breaks page when it closes)

    @override
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "body":
            self._skip_depth = 0  # an unclosed <head> ends at <body>
        if tag in SKIP_TAGS:
            self._skip_depth += 1
            return
        breaks = {b.lower() for b in PAGE_BREAK.findall(dict(attrs).get("style") or "")}
        if tag == "tr":
            if self._cells is not None:
                self.flush_row()  # old HTML often omits </tr>
            self.flush()
            self._cells = []
        elif tag in CELL_TAGS and self._cells is not None:
            self._cells.append([])
        elif tag in BLOCK_TAGS:
            if self._cells is not None:
                self._cell_text(" ")  # a block inside a cell is just a space
            else:
                self.flush()
        if "before" in breaks:
            self.page_break()
        if tag in VOID_TAGS:
            if "after" in breaks:
                self.page_break()
        else:
            self._open.append((tag, "after" in breaks))

    @override
    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag in ("tr", "table") and self._cells is not None:
            self.flush_row()
        elif tag in BLOCK_TAGS and self._cells is None:
            self.flush()
        # Pop up to the matching open tag (tolerates unclosed tags in old HTML).
        if any(open_tag == tag for open_tag, _ in self._open):
            while self._open:
                open_tag, break_after = self._open.pop()
                if break_after:
                    self.page_break()
                if open_tag == tag:
                    break

    @override
    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._cells is not None:
            self._cell_text(data)
        else:
            self._buffer.append(data)

    def _cell_text(self, text: str) -> None:
        assert self._cells is not None
        if not self._cells:
            self._cells.append([])  # text in a <tr> before any <td>
        self._cells[-1].append(text)

    def flush(self) -> None:
        line = clean("".join(self._buffer))
        self._buffer = []
        if line:
            self.lines.append(line)

    def flush_row(self) -> None:
        assert self._cells is not None
        cells = [clean("".join(cell)) for cell in self._cells]
        self._cells = None
        row = " | ".join(c for c in cells if c)
        if row:
            self.lines.append(row)

    def page_break(self) -> None:
        self.flush()
        self.page += 1
        self.lines.append(f"[Page {self.page}]")

    def finish(self) -> list[str]:
        self.close()
        if self._cells is not None:
            self.flush_row()
        self.flush()
        while len(self.lines) > 1 and PAGE_MARKER.fullmatch(self.lines[-1]):
            self.lines.pop()  # a trailing page break has no content after it
        return self.lines


def html_to_lines(html: str) -> list[str]:
    parser = _Parser()
    parser.feed(html)
    return parser.finish()


def text_to_lines(text: str) -> list[str]:
    """Plain-text exhibits: join wrapped lines into paragraphs; <PAGE> starts a page."""
    lines = ["[Page 1]"]
    page = 1
    paragraph: list[str] = []

    def flush() -> None:
        line = clean(" ".join(paragraph))
        paragraph.clear()
        if line:
            lines.append(line)

    for raw_line in text.splitlines():
        if PAGE_TAG.match(raw_line):
            flush()
            page += 1
            lines.append(f"[Page {page}]")
        elif raw_line.strip():
            paragraph.append(raw_line)
        else:
            flush()
    flush()
    return lines


def decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def extract_file(src: Path) -> str:
    text = decode(src.read_bytes())
    lines = text_to_lines(text) if src.suffix.lower() == ".txt" else html_to_lines(text)
    return "\n".join(lines) + "\n"


def write_index(manifest: Manifest, text_dir: Path) -> Path:
    """Write `<text_dir>/_index.md`: what the agent can't read from the documents.

    One row per document: file, title, legal date, what it amends, line and page
    counts. Facts only, no hints about answers.
    """
    rows = [
        f"# Documents: {manifest.company}",
        "",
        "Generated from data/manifest.yaml. `dated` is each document's legal "
        '"dated as of" date and gives their order; filing dates do not.',
        "",
        "| file | title | dated | amends | lines | pages |",
        "|---|---|---|---|---|---|",
    ]
    for doc in manifest.documents:
        lines = (text_dir / f"{doc.id}.txt").read_text(encoding="utf-8").splitlines()
        pages = sum(1 for line in lines if PAGE_MARKER.fullmatch(line))
        rows.append(
            f"| {doc.id}.txt | {doc.title} | {doc.dated} | {doc.amends or '-'} "
            f"| {len(lines)} | {pages} |"
        )
    path = text_dir / INDEX_NAME
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def extract_all(manifest: Manifest, raw_dir: Path, text_dir: Path) -> list[Path]:
    """Write `<text_dir>/<id>.txt` for every document (always overwriting) and the index."""
    text_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for doc in manifest.documents:
        src = raw_path(raw_dir, doc.id, doc.url)
        if not src.exists():
            raise FileNotFoundError(
                f"{src} is missing; run `amendment-agent fetch` first"
            )
        dest = text_dir / f"{doc.id}.txt"
        content = extract_file(src)
        dest.write_text(content, encoding="utf-8")
        pages = sum(1 for line in content.splitlines() if PAGE_MARKER.fullmatch(line))
        print(
            f"extract {doc.id}: {dest} ({len(content.splitlines()):,} lines, {pages} pages)"
        )
        paths.append(dest)
    print(f"index   {write_index(manifest, text_dir)}")
    return paths
