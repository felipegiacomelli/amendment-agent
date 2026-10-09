"""The agent's tools over a folder of files. USER-OWNED: the scaffold defines the contract only.

Reads stay inside data/text/ and writes inside the run's output directory. A path
outside its root returns an error result the model can see; tools never raise.
"""

from pathlib import Path

import attrs
from anthropic.types import ToolParam

from amendment_agent.settings import OUTPUT_DIR, TEXT_DIR


@attrs.frozen
class ToolResult:
    text: str
    is_error: bool


# Names, descriptions and input schemas are prompt engineering: user-owned.
TOOL_SCHEMAS: list[ToolParam] = []


def list_files(path: str = "", *, root: Path = TEXT_DIR) -> ToolResult:
    """List the documents under `root/path`."""
    raise NotImplementedError


def grep(pattern: str, path: str = "", *, root: Path = TEXT_DIR) -> ToolResult:
    """Search files under `root/path`; report matches with 1-based line numbers."""
    raise NotImplementedError


def read_file(
    path: str, start_line: int, end_line: int, *, root: Path = TEXT_DIR
) -> ToolResult:
    """Return lines `start_line..end_line` (1-based, inclusive) of `root/path`."""
    raise NotImplementedError


def write_output(filename: str, content: str, *, root: Path = OUTPUT_DIR) -> ToolResult:
    """Write `content` to `root/filename`; the loop passes the run's output_dir as root."""
    raise NotImplementedError


def execute_tool(
    name: str, tool_input: dict[str, object], *, output_dir: Path
) -> ToolResult:
    """Run the named tool. Unknown names and invalid input are error results."""
    raise NotImplementedError
