import inspect
from pathlib import Path

import pytest

from amendment_agent import tools
from amendment_agent.schema import ANSWER_FILENAME, Citation, load_answer
from amendment_agent.tools import (
    citation_problem,
    execute_tool,
    grep,
    list_files,
    read_file,
    write_answer,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    text = tmp_path / "text"
    text.mkdir()
    (text / "original.txt").write_text(
        "[Page 1]\nCREDIT AGREEMENT\nApplicable Rate means 1.25%\n", encoding="utf-8"
    )
    (text / "amendment-1.txt").write_text(
        "[Page 1]\nThe Applicable Rate is amended to 1.50%\n", encoding="utf-8"
    )
    (tmp_path / "secret.txt").write_text("applicable rate leak\n", encoding="utf-8")
    return text


def test_list_files(root: Path) -> None:
    result = list_files(root=root)
    assert not result.is_error
    assert result.text.splitlines() == ["amendment-1.txt", "original.txt"]


def test_list_files_rejects_escape(root: Path) -> None:
    assert list_files("..", root=root).is_error


def test_grep_line_numbers_match_read_file(root: Path) -> None:
    result = grep("applicable rate", root=root)
    assert not result.is_error
    assert result.text.splitlines() == [
        "amendment-1.txt:2: The Applicable Rate is amended to 1.50%",
        "original.txt:3: Applicable Rate means 1.25%",
    ]
    assert (
        "3: Applicable Rate means 1.25%"
        in read_file("original.txt", 3, 3, root=root).text
    )


def test_grep_single_file_and_no_matches(root: Path) -> None:
    assert grep("rate", "original.txt", root=root).text.startswith("original.txt:3:")
    assert grep("nothing here", root=root).text == "(no matches)"


def test_grep_errors(root: Path) -> None:
    assert grep("(", root=root).is_error
    assert grep("rate", "../secret.txt", root=root).is_error
    assert grep("rate", "missing.txt", root=root).is_error


def test_grep_caps_hits_and_long_lines(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools, "MAX_GREP_HITS", 2)
    monkeypatch.setattr(tools, "MAX_HIT_CHARS", 10)
    lines = grep("a", root=root).text.splitlines()
    assert len(lines) == 3
    assert lines[1].endswith("...")  # lines[0] is the short "[Page 1]"
    assert lines[-1].startswith("(stopped at 2 hits")


def test_read_file_range(root: Path) -> None:
    result = read_file("original.txt", 2, 99, root=root)
    assert not result.is_error
    assert result.text.splitlines() == [
        "original.txt lines 2-3 of 3",
        "2: CREDIT AGREEMENT",
        "3: Applicable Rate means 1.25%",
    ]


@pytest.mark.parametrize(("start", "end"), [(0, 2), (3, 2), (4, 5)])
def test_read_file_invalid_range(root: Path, start: int, end: int) -> None:
    result = read_file("original.txt", start, end, root=root)
    assert result.is_error
    assert "has lines 1-3" in result.text


def test_read_file_caps_lines(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "MAX_READ_LINES", 2)
    result = read_file("original.txt", 1, 3, root=root)
    assert result.text.splitlines()[0] == (
        "original.txt lines 1-2 of 3 (capped at 2 lines per call)"
    )


def test_read_file_rejects_escape(root: Path) -> None:
    assert read_file("../secret.txt", 1, 1, root=root).is_error


# A credit-agreement-style document: TOC row, headings with and without the period
# extraction sometimes drops, curly quotes, wrapped text that looks like a number.
AGREEMENT = """[Page 1]
7.11 | Burdensome Agreements | 101
7.10 Investments. Make any Investments, except as permitted.
7.11 Burdensome Agreements. Enter into any agreement that limits the “Company”.
The Consolidated Leverage Ratio shall not exceed 3.75 to 1.00 at any time.
9.10 of the Credit Agreement; provided further, that it survives.
9.06 Resignation of Administrative Agent
(a) The Administrative Agent may resign at any time.
"""
AMENDMENT = """[Page 1]
1.Amendments to Credit Agreement. The Credit Agreement is amended as follows:
(a) Section 7.11 is amended to read 4.50 to 1.00 in its entirety.
"""
CITATION: dict[str, object] = {
    "document": "original",
    "section": "Section 7.11(a)",
    "lines": [4, 5],
    "quote": "shall not exceed 3.75 to 1.00",
}


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    text = tmp_path / "corpus"
    text.mkdir()
    (text / "original.txt").write_text(AGREEMENT, encoding="utf-8")
    (text / "amendment-1.txt").write_text(AMENDMENT, encoding="utf-8")
    return text


def test_write_answer_writes_a_valid_answer_file(corpus: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = write_answer(
        "answered", "Covenant in 7.11.", [CITATION], "3.75:1", root=out, text_dir=corpus
    )
    assert not result.is_error, result.text
    answer = load_answer(out / ANSWER_FILENAME)
    assert answer.value == "3.75:1"
    assert answer.citations[0].lines == (4, 5)


@pytest.mark.parametrize(
    "citation",
    [
        # curly quotes and spacing in the file don't have to be copied exactly
        {**CITATION, "lines": [4, 4], "quote": 'limits the "Company".'},
        {**CITATION, "quote": "the  “Company”. The Consolidated   Leverage"},
        # a heading whose period was lost in extraction still counts
        {
            **CITATION,
            "section": "9.06",
            "lines": [8, 8],
            "quote": "Administrative Agent may resign",
        },
        # wrapped text starting with "9.10 of" is not a heading: still 7.11
        {**CITATION, "lines": [6, 6], "quote": "provided further, that it survives"},
        # an amendment's own sections are numbered "1.", "2."
        {
            "document": "amendment-1",
            "section": "1(a)",
            "lines": [3, 3],
            "quote": "amended to read 4.50 to 1.00",
        },
    ],
)
def test_citation_problem_accepts(corpus: Path, citation: dict[str, object]) -> None:
    assert citation_problem(Citation.model_validate(citation), corpus) is None


@pytest.mark.parametrize(
    ("citation", "problem"),
    [
        ({**CITATION, "document": "missing"}, "no document"),
        ({**CITATION, "document": "../secret"}, "no document"),
        ({**CITATION, "lines": [4, 99]}, "past its end"),
        ({**CITATION, "quote": "4.50 to 1.00"}, "under 20 characters"),
        ({**CITATION, "quote": "shall not exceed 4.50 to 1.00"}, "quote not found"),
        (
            {
                **CITATION,
                "lines": [5, 5],
                "quote": "any agreement that limits the Company",
            },
            "not found",
        ),
        ({**CITATION, "section": "7.10"}, "in section 7.11"),
        ({**CITATION, "lines": [3, 5], "section": "7.10"}, "into 7.11"),
        ({**CITATION, "section": "Burdensome Agreements"}, "in section 7.11"),
        (
            {**CITATION, "lines": [1, 2], "quote": "Burdensome Agreements | 101"},
            "not under",
        ),
    ],
)
def test_citation_problem_rejects(
    corpus: Path, citation: dict[str, object], problem: str
) -> None:
    found = citation_problem(Citation.model_validate(citation), corpus)
    assert found is not None and problem in found


def test_citation_problem_caps_range_width(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools, "MAX_CITATION_LINES", 1)
    found = citation_problem(Citation.model_validate(CITATION), corpus)
    assert found is not None and "cite at most 1" in found


def test_write_answer_insufficient_evidence(tmp_path: Path) -> None:
    result = write_answer(
        "insufficient_evidence", "Not in the documents.", [], root=tmp_path
    )
    assert not result.is_error
    assert load_answer(tmp_path / ANSWER_FILENAME).value is None


@pytest.mark.parametrize(
    ("status", "citations", "value"),
    [
        ("answered", [CITATION], None),  # answered needs a value
        ("answered", [], "3.75:1"),  # answered needs a citation
        ("answered", [{**CITATION, "lines": [4]}], "3.75:1"),  # lines needs two items
        ("answered", [{**CITATION, "lines": [5, 4]}], "3.75:1"),  # end before start
        ("answered", [{**CITATION, "quote": None}], "3.75:1"),  # quote is required
        ("maybe", [], None),  # unknown status
        ("answered", [{**CITATION, "section": "7.10"}], "3.75:1"),  # citation_problem
    ],
)
def test_write_answer_rejects_invalid_answers(
    corpus: Path,
    tmp_path: Path,
    status: str,
    citations: list[dict[str, object]],
    value: str | None,
) -> None:
    result = write_answer(
        status, "why", citations, value, root=tmp_path, text_dir=corpus
    )
    assert result.is_error
    assert "answer not written" in result.text
    assert not (tmp_path / ANSWER_FILENAME).exists()


def test_execute_tool_passes_output_dir(tmp_path: Path) -> None:
    result = execute_tool(
        "write_answer",
        {"status": "insufficient_evidence", "explanation": "none", "citations": []},
        output_dir=tmp_path,
    )
    assert not result.is_error
    assert (tmp_path / ANSWER_FILENAME).exists()


def test_execute_tool_errors(tmp_path: Path) -> None:
    assert execute_tool("rm", {}, output_dir=tmp_path).is_error
    assert execute_tool("read_file", {"path": "x"}, output_dir=tmp_path).is_error
    smuggled: dict[str, object] = {"path": "", "root": "/"}
    assert execute_tool("list_files", smuggled, output_dir=tmp_path).is_error
    smuggled = {"status": "x", "explanation": "", "citations": [], "text_dir": "/"}
    assert execute_tool("write_answer", smuggled, output_dir=tmp_path).is_error


def test_schemas_match_functions() -> None:
    assert [s["name"] for s in tools.TOOL_SCHEMAS] == list(tools.TOOL_MAP)
    for schema in tools.TOOL_SCHEMAS:
        signature = inspect.signature(tools.TOOL_MAP[schema["name"]])
        # keyword-only parameters (root, text_dir) are set by the loop, not the model
        params = {
            name: p
            for name, p in signature.parameters.items()
            if p.kind is not inspect.Parameter.KEYWORD_ONLY
        }
        required_args = {
            name for name, p in params.items() if p.default is inspect.Parameter.empty
        }
        properties = schema["input_schema"]["properties"]
        required = schema["input_schema"]["required"]
        assert isinstance(properties, dict) and isinstance(required, list)
        assert set(properties) == set(params)
        assert set(required) == required_args
