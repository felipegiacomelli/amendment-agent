import copy
import json
import time
from pathlib import Path
from typing import Any, cast

import httpx2  # the SDK builds its errors on httpx2, not the project's httpx
import pytest
from anthropic import (
    Anthropic,
    APIConnectionError,
    BadRequestError,
    RateLimitError,
)
from anthropic.types import Message

from amendment_agent import agent
from amendment_agent.agent import StopReason, run_agent
from amendment_agent.recorder import RunRecorder, read_events
from amendment_agent.schema import ANSWER_FILENAME
from amendment_agent.tools import ToolResult

MODEL = "claude-sonnet-5-5"
USAGE = {"input_tokens": 100, "output_tokens": 10}  # $0.0003 on Sonnet 5.5
REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
ANSWER = {
    "status": "answered",
    "value": "2.50:1",
    "explanation": "e",
    "citations": [
        {"document": "original", "section": "6.12", "lines": [1, 1], "quote": "q"}
    ],
}

type Block = dict[str, Any]


def message(
    stop_reason: str, *content: Block, usage: dict[str, int] = USAGE
) -> Message:
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": list(content),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage,
        }
    )


def text(value: str = "thinking aloud") -> Block:
    return {"type": "text", "text": value}


def tool_use(id: str, name: str = "grep", **tool_input: Any) -> Block:
    return {"type": "tool_use", "id": id, "name": name, "input": tool_input}


def calls_tool(id: str, name: str = "grep", **tool_input: Any) -> Message:
    return message("tool_use", tool_use(id, name, **tool_input))


def ends() -> Message:
    return message("end_turn", text("done"))


class FakeClient:
    """`messages.create` returns (or raises) the scripted items in order and
    records a deep copy of every request's params."""

    def __init__(self, *script: Message | Exception) -> None:
        self.script = list(script)
        self.params: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **params: Any) -> Message:
        self.params.append(copy.deepcopy(params))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeTools:
    """Stands in for execute_tool: records calls, returns scripted results by tool
    name (default ok), and writes `answer` as answer.json on write_answer."""

    def __init__(
        self, results: dict[str, ToolResult] | None = None, answer: str = ""
    ) -> None:
        self.results = results or {}
        self.answer = answer
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(
        self, name: str, tool_input: dict[str, object], *, output_dir: Path
    ) -> ToolResult:
        self.calls.append((name, tool_input))
        if name == "write_answer":
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / ANSWER_FILENAME).write_text(self.answer, encoding="utf-8")
        return self.results.get(name, ToolResult(text=f"{name} ok", is_error=False))


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch) -> FakeTools:
    fake = FakeTools(answer=json.dumps(ANSWER))
    monkeypatch.setattr(agent, "execute_tool", fake)
    return fake


@pytest.fixture
def delays(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    return slept


def run(
    tmp_path: Path,
    client: FakeClient,
    *,
    max_turns: int = 10,
    max_cost_usd: float = 1.0,
) -> tuple[StopReason, list[dict[str, Any]]]:
    recorder = RunRecorder(tmp_path / "runs", MODEL, "Q?", text_dir=tmp_path / "text")
    stop = run_agent(
        "Q?",
        client=cast(Anthropic, client),
        model=MODEL,
        recorder=recorder,
        output_dir=tmp_path / "out",
        max_turns=max_turns,
        max_cost_usd=max_cost_usd,
    )
    return stop, read_events(recorder.path)


def results_in(request: dict[str, Any]) -> list[dict[str, Any]]:
    """The tool results in the last user message of a request."""
    last = request["messages"][-1]
    assert last["role"] == "user"
    return cast(list[dict[str, Any]], last["content"])


def test_history_grows_by_the_assistant_turn_as_received_and_its_results(
    tmp_path: Path, tools: FakeTools
) -> None:
    thinking = {"type": "thinking", "thinking": "hmm", "signature": "sig-123"}
    first = message("tool_use", thinking, text(), tool_use("t1", pattern="Rate"))
    client = FakeClient(first, ends())

    run(tmp_path, client)

    before, after = (p["messages"] for p in client.params)
    assert before == [{"role": "user", "content": "Q?"}]
    assert after[:1] == before
    assert after[1] == {"role": "assistant", "content": first.content}
    assert after[1]["content"][0].signature == "sig-123"
    assert after[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "t1",
                "content": "grep ok",
                "is_error": False,
            }
        ],
    }
    assert len(after) == 3


def test_parallel_calls_get_all_results_in_one_message_errors_included(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = ToolResult(text="line 9000 is past the end (4212 lines)", is_error=True)
    tools = FakeTools({"read_file": error})
    monkeypatch.setattr(agent, "execute_tool", tools)
    both = message(
        "tool_use", tool_use("t1", pattern="x"), tool_use("t2", "read_file", start=9000)
    )
    client = FakeClient(both, ends())

    stop, _ = run(tmp_path, client)

    assert len(client.params) == 2  # no request between the two results
    assert [
        (r["tool_use_id"], r["is_error"]) for r in results_in(client.params[1])
    ] == [
        ("t1", False),
        ("t2", True),
    ]
    assert results_in(client.params[1])[1]["content"] == error.text
    assert stop is StopReason.NO_ANSWER


def test_turn_limit_sends_exactly_max_turns_requests(
    tmp_path: Path, tools: FakeTools
) -> None:
    client = FakeClient(*(calls_tool(f"t{i}", pattern=str(i)) for i in range(5)))

    stop, _ = run(tmp_path, client, max_turns=3)

    assert stop is StopReason.MAX_TURNS
    assert len(client.params) == 3


def test_cost_limit_stops_before_the_next_request(
    tmp_path: Path, tools: FakeTools
) -> None:
    client = FakeClient(calls_tool("t1", pattern="x"), ends())

    stop, events = run(tmp_path, client, max_cost_usd=0.0001)

    assert stop is StopReason.MAX_COST
    assert len(client.params) == 1
    assert events[-1]["stop_reason"] == "max_cost"


def test_a_repeated_call_is_refused_and_the_third_ends_the_run(
    tmp_path: Path, tools: FakeTools
) -> None:
    client = FakeClient(*(calls_tool(f"t{i}", pattern="Rate") for i in range(3)))

    stop, _ = run(tmp_path, client)

    assert stop is StopReason.REPEATED_CALL
    assert tools.calls == [("grep", {"pattern": "Rate"})]  # executed once
    refused = results_in(client.params[2])[0]
    assert refused["is_error"] and "already made this exact call" in refused["content"]


def test_a_revisit_with_calls_in_between_is_refused_but_the_run_goes_on(
    tmp_path: Path, tools: FakeTools
) -> None:
    read = {"path": "original.txt", "start_line": 4200, "end_line": 4220}
    client = FakeClient(
        calls_tool("t1", "read_file", **read),
        calls_tool("t2", "read_file", path="amendment-1.txt"),
        calls_tool("t3", "read_file", **read),
        ends(),
    )

    stop, _ = run(tmp_path, client)

    assert stop is StopReason.NO_ANSWER
    assert len(client.params) == 4
    assert [name for name, _ in tools.calls] == ["read_file", "read_file"]
    assert results_in(client.params[3])[0]["is_error"]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (message("refusal"), StopReason.REFUSAL),
        (message("max_tokens", text("cut off")), StopReason.MAX_TOKENS),
        (ends(), StopReason.NO_ANSWER),
        (message("pause_turn", text()), StopReason.PROVIDER_ERROR),
        (message("stop_sequence", text()), StopReason.PROVIDER_ERROR),
    ],
)
def test_each_ending_has_its_own_stop_reason(
    tmp_path: Path, tools: FakeTools, response: Message, expected: StopReason
) -> None:
    stop, events = run(tmp_path, FakeClient(response))

    assert stop is expected
    assert events[-1]["event"] == "run_end"
    assert events[-1]["stop_reason"] == expected.value


def test_answered_needs_a_valid_answer_file_written_this_run(
    tmp_path: Path, tools: FakeTools
) -> None:
    client = FakeClient(calls_tool("t1", "write_answer"), ends())

    stop, _ = run(tmp_path, client)

    assert stop is StopReason.ANSWERED


def test_an_invalid_answer_file_is_not_answered(
    tmp_path: Path, tools: FakeTools
) -> None:
    tools.answer = json.dumps(ANSWER | {"citations": []})  # answered needs one
    client = FakeClient(calls_tool("t1", "write_answer"), ends())

    stop, _ = run(tmp_path, client)

    assert stop is StopReason.NO_ANSWER


def test_retryable_errors_are_retried_honouring_retry_after(
    tmp_path: Path, tools: FakeTools, delays: list[float]
) -> None:
    limited = httpx2.Response(429, request=REQUEST, headers={"retry-after": "7"})
    client = FakeClient(
        APIConnectionError(request=REQUEST),
        RateLimitError("slow down", response=limited, body=None),
        ends(),
    )

    stop, events = run(tmp_path, client)

    assert stop is StopReason.NO_ANSWER
    assert delays == [1.0, 7.0]
    requests = [e for e in events if e["event"] == "model_request"]
    assert [e["turn"] for e in requests] == [0, 0, 0]  # each retry is a request
    assert events[-1]["cost_complete"] is False  # two requests got no response


def test_exhausted_retries_end_the_run_as_a_provider_error(
    tmp_path: Path, tools: FakeTools, delays: list[float]
) -> None:
    client = FakeClient(*(APIConnectionError(request=REQUEST) for _ in range(4)))

    stop, events = run(tmp_path, client)

    assert stop is StopReason.PROVIDER_ERROR
    assert len(client.params) == agent.MAX_RETRIES + 1
    assert delays == [1.0, 2.0, 4.0]
    assert not [e for e in events if e["event"] == "tool_call"]  # model never sees it
    assert events[-1]["stop_reason"] == "provider_error"


def test_a_non_retryable_error_ends_the_run_at_once(
    tmp_path: Path, tools: FakeTools, delays: list[float]
) -> None:
    bad = httpx2.Response(400, request=REQUEST)
    client = FakeClient(BadRequestError("bad", response=bad, body=None))

    stop, _ = run(tmp_path, client)

    assert stop is StopReason.PROVIDER_ERROR
    assert len(client.params) == 1
    assert delays == []


def test_a_refusal_on_the_call_that_crosses_the_cost_limit_stays_a_refusal(
    tmp_path: Path, tools: FakeTools
) -> None:
    stop, _ = run(tmp_path, FakeClient(message("refusal")), max_cost_usd=0.0001)

    assert stop is StopReason.REFUSAL
