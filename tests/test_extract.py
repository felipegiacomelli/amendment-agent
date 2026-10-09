from collections.abc import Callable
from pathlib import Path

import pytest

from amendment_agent.extract import (
    extract_all,
    extract_file,
    html_to_lines,
    text_to_lines,
)
from amendment_agent.manifest import Manifest

type MakeManifest = Callable[..., Manifest]  # the conftest fixture's type


def test_blocks_become_lines_and_head_script_style_are_dropped() -> None:
    html = (
        "<html><head><title>T</title><style>p {}</style></head><body>"
        "<p>First   paragraph\n wrapped.</p><div>Second <b>bold</b> block</div>"
        "<script>var x = 1;</script></body></html>"
    )
    assert html_to_lines(html) == [
        "[Page 1]",
        "First paragraph wrapped.",
        "Second bold block",
    ]


def test_page_break_markers_before_and_after() -> None:
    html = (
        "<p>Cover</p>"
        '<p style="page-break-before: always">Body</p>'
        '<div style="PAGE-BREAK-AFTER:always"><div>Inner one</div><div>Inner two</div></div>'
        "<p>Three</p>"
        '<hr style="page-break-after:always">'
    )
    # The marker after the outer div comes when *it* closes, not at the first inner
    # </div>. A trailing marker with no content after it is dropped.
    assert html_to_lines(html) == [
        "[Page 1]",
        "Cover",
        "[Page 2]",
        "Body",
        "Inner one",
        "Inner two",
        "[Page 3]",
        "Three",
    ]


def test_table_rows_join_non_empty_cells() -> None:
    html = (
        "<table>"
        "<tr><td><p>Pricing Level</p></td><td>&nbsp;</td><td>Debt<br>Rating</td></tr>"
        "<tr><td>1</td><td></td><td>A-</td></tr>"
        "</table>"
    )
    assert html_to_lines(html) == ["[Page 1]", "Pricing Level | Debt Rating", "1 | A-"]


def test_entities_nbsp_and_zero_width_spaces() -> None:
    html = "<p>Section&nbsp;7.06&#8203; &ldquo;Consolidated&rdquo; &amp; Co.</p>"
    assert html_to_lines(html) == ["[Page 1]", "Section 7.06 “Consolidated” & Co."]


def test_cp1252_bytes_are_decoded(tmp_path: Path) -> None:
    src = tmp_path / "doc.htm"
    src.write_bytes("<p>Borrower’s rate</p>".encode("cp1252"))
    assert extract_file(src) == "[Page 1]\nBorrower’s rate\n"


def test_plain_text_joins_paragraphs_and_marks_pages() -> None:
    text = "CREDIT AGREEMENT\n\nSection 1.01  Defined\n   Terms.\n<PAGE>\nPage two.\n"
    assert text_to_lines(text) == [
        "[Page 1]",
        "CREDIT AGREEMENT",
        "Section 1.01 Defined Terms.",
        "[Page 2]",
        "Page two.",
    ]


def test_extract_all_writes_one_text_file_per_document(
    tmp_path: Path, make_manifest: MakeManifest
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "original.htm").write_text("<p>Hi</p>", encoding="utf-8")
    manifest = make_manifest("https://www.sec.gov/Archives/edgar/data/1/a.htm")
    paths = extract_all(manifest, raw, tmp_path / "text")
    assert paths == [tmp_path / "text" / "original.txt"]
    assert paths[0].read_text(encoding="utf-8") == "[Page 1]\nHi\n"


def test_extract_all_writes_a_factual_index(
    tmp_path: Path, make_manifest: MakeManifest
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "original.htm").write_text(
        "<p>One</p><p style='page-break-before:always'>Two</p>", encoding="utf-8"
    )
    (raw / "amendment-1.htm").write_text("<p>Amend</p>", encoding="utf-8")
    manifest = make_manifest(
        "https://www.sec.gov/Archives/edgar/data/1/a.htm",
        "https://www.sec.gov/Archives/edgar/data/1/b.htm",
    )
    extract_all(manifest, raw, tmp_path / "text")
    index = (tmp_path / "text" / "_index.md").read_text(encoding="utf-8")
    assert "| original.txt | Document | 2020-01-01 | - | 4 | 2 |" in index
    assert "| amendment-1.txt | Document | 2020-01-01 | original | 2 | 1 |" in index


def test_extract_all_missing_raw_file_points_to_fetch(
    tmp_path: Path, make_manifest: MakeManifest
) -> None:
    manifest = make_manifest("https://www.sec.gov/Archives/edgar/data/1/a.htm")
    with pytest.raises(FileNotFoundError, match="amendment-agent fetch"):
        extract_all(manifest, tmp_path / "raw", tmp_path / "text")


def test_unclosed_tr_and_td_rows() -> None:
    html = "<table><tr><td>A</td><td>&nbsp;</td><td>B<tr><td>1<td>&#160;<td>2</table>"
    assert html_to_lines(html) == ["[Page 1]", "A | B", "1 | 2"]


def test_unclosed_head_ends_at_body() -> None:
    html = "<html><head><title>T</title><body><p>Body text</p></body></html>"
    assert html_to_lines(html) == ["[Page 1]", "Body text"]
