"""Append-only JSONL transcript of one agent run: requests, responses, tools, cost.

The agent loop calls this explicitly. Every event is flushed as it happens, so a
crashed or hung run still leaves its transcript up to that point.
"""

import datetime as dt
import hashlib
import json
import secrets
import subprocess
import traceback
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import attrs
from anthropic.types import Message, MessageParam, Usage
from pydantic import BaseModel

from amendment_agent.settings import TEXT_DIR


@attrs.frozen
class Price:
    """USD per million tokens."""

    input: float
    output: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float


PRICES_SOURCE = (
    "https://platform.claude.com/docs/en/about-claude/pricing, checked 2026-10-09; "
    "global routing (no inference_geo multiplier)"
)
# Only models priced flat across the whole context window. Haiku 5.5 is left out on
# purpose: prompts over 100k tokens are billed at a higher tier this table can't express.
PRICES: dict[str, Price] = {
    "claude-sonnet-5-5": Price(
        input=2.00,
        output=10.00,
        cache_write_5m=2.50,
        cache_write_1h=4.00,
        cache_read=0.10,
    ),
    "claude-opus-5-5": Price(
        input=4.00,
        output=20.00,
        cache_write_5m=5.00,
        cache_write_1h=8.00,
        cache_read=0.20,
    ),
}

TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def cost(model: str, usage: Usage) -> float:
    """USD for one model call. Thinking tokens are already inside `output_tokens`."""
    price = PRICES[model]
    if usage.cache_creation is not None:
        writes = (
            usage.cache_creation.ephemeral_5m_input_tokens * price.cache_write_5m
            + usage.cache_creation.ephemeral_1h_input_tokens * price.cache_write_1h
        )
    else:
        writes = (usage.cache_creation_input_tokens or 0) * price.cache_write_5m
    return (
        usage.input_tokens * price.input
        + writes
        + (usage.cache_read_input_tokens or 0) * price.cache_read
        + usage.output_tokens * price.output
    ) / 1_000_000


def new_id(label: str) -> str:
    """Unique, sortable id: UTC timestamp, label, 6 random hex characters."""
    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{stamp}-{label}-{secrets.token_hex(3)}"


def git_revision() -> dict[str, object]:
    """The code that produced a run: commit and whether the tree had changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def fingerprint(directory: Path) -> dict[str, str]:
    """sha256 of every file directly in `directory`; empty if it doesn't exist."""
    if not directory.is_dir():
        return {}
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _jsonable(value: Any) -> Any:
    """Messages mix plain dicts with SDK blocks appended as-is; make them JSON-safe."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds")


class RunRecorder:
    """Writes `runs_dir/<run_id>.jsonl`, one JSON event per line."""

    def __init__(
        self,
        runs_dir: Path,
        model: str,
        question: str,
        *,
        label: str = "ask",
        text_dir: Path = TEXT_DIR,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if model not in PRICES:
            raise ValueError(
                f"no price for model {model!r}; supported: {', '.join(sorted(PRICES))}"
            )
        self.model = model
        self.turns = 0
        self.total_cost_usd = 0.0
        self._tokens = dict.fromkeys(TOKEN_FIELDS, 0)
        self._pending_request = False
        self._lost_response = False
        self._ended = False
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = new_id(label)
        self.path = runs_dir / f"{self.run_id}.jsonl"
        self._file = self.path.open("x", encoding="utf-8")  # never overwrite a run
        self._write(
            "run_start",
            run_id=self.run_id,
            label=label,
            model=model,
            question=question,
            git=git_revision(),
            corpus=fingerprint(text_dir),
            prices=attrs.asdict(PRICES[model]) | {"source": PRICES_SOURCE},
            config=extra or {},
        )

    @property
    def cost_complete(self) -> bool:
        """False once any request was sent with no response recorded: cost unknown, not zero."""
        return not (self._pending_request or self._lost_response)

    def model_request(self, turn: int, params: dict[str, Any]) -> None:
        """Call just before `client.messages.create(**params)`.

        Logs the request exactly as sent, messages included, so a failed run still
        shows what the model received.
        """
        logged = {key: _jsonable(value) for key, value in params.items()}
        self._lost_response |= self._pending_request  # earlier request never answered
        self._pending_request = True
        self._write("model_request", turn=turn, params=logged)

    def model_call(self, turn: int, response: Message) -> float:
        """Record one response verbatim (thinking included); return its cost."""
        call_cost = cost(self.model, response.usage)
        self._pending_request = False
        self.turns += 1
        self.total_cost_usd += call_cost
        for field in TOKEN_FIELDS:
            self._tokens[field] += getattr(response.usage, field) or 0
        self._write(
            "model_call",
            turn=turn,
            response=response.model_dump(mode="json", exclude_none=True),
            cost_usd=call_cost,
            total_cost_usd=self.total_cost_usd,
        )
        return call_cost

    def tool_call(
        self,
        turn: int,
        tool_use_id: str,
        name: str,
        tool_input: dict[str, object],
        result: str,
        is_error: bool,
    ) -> None:
        self._write(
            "tool_call",
            turn=turn,
            tool_use_id=tool_use_id,
            name=name,
            input=tool_input,
            result=result,
            is_error=is_error,
        )

    def end(self, stop_reason: str, messages: list[MessageParam]) -> None:
        """Write totals and the full message list. A second call is a no-op."""
        if self._ended:
            return
        self._write(
            "run_end",
            stop_reason=stop_reason,
            **self._totals(),
            messages=_jsonable(messages),
        )
        self._ended = True
        self._file.close()

    def _totals(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "cost_usd": self.total_cost_usd,
            "cost_complete": self.cost_complete,
            **self._tokens,
        }

    def _write(self, event: str, **data: Any) -> None:
        line = json.dumps({"ts": _now(), "event": event, **data}, default=str)
        self._file.write(line + "\n")
        self._file.flush()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self._ended:
            if exc is not None:
                error = "".join(traceback.format_exception(exc))
                self._write(
                    "run_end", stop_reason="error", **self._totals(), error=error
                )
            else:
                self._write("run_end", stop_reason="not_ended", **self._totals())
            self._ended = True
        self._file.close()  # closing twice is harmless
