import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Message, MessageParam

from amendment_agent.recorder import PRICES, RunRecorder, cost

MODEL = "claude-sonnet-5-5"
USAGE: dict[str, Any] = {
    "input_tokens": 1000,
    "output_tokens": 200,
    "cache_creation_input_tokens": 3000,
    "cache_read_input_tokens": 10000,
}
# Sonnet 5.5: 1000*2 + 3000*2.50 + 10000*0.10 + 200*10 = 12_500 per million.
USAGE_COST = 0.0125


def response(
    stop_reason: str = "end_turn",
    content: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> Message:
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": content or [{"type": "text", "text": "done"}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage or USAGE,
        }
    )


def events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_cost_uses_the_verified_price_table() -> None:
    assert cost(MODEL, response().usage) == pytest.approx(USAGE_COST)


def test_cost_prices_1h_cache_writes_from_the_breakdown() -> None:
    breakdown = {"ephemeral_5m_input_tokens": 2000, "ephemeral_1h_input_tokens": 1000}
    usage = response(usage=USAGE | {"cache_creation": breakdown}).usage
    # 1000*2 + (2000*2.50 + 1000*4.00) + 10000*0.10 + 200*10 = 14_000 per million.
    assert cost(MODEL, usage) == pytest.approx(0.014)


def test_cost_treats_missing_cache_fields_as_zero() -> None:
    usage = response(usage={"input_tokens": 1_000_000, "output_tokens": 0}).usage
    assert cost(MODEL, usage) == pytest.approx(PRICES[MODEL].input)


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "gpt-x"])
def test_unpriced_models_are_rejected_before_any_file_is_written(
    tmp_path: Path, model: str
) -> None:
    with pytest.raises(ValueError, match=f"no price for model '{model}'"):
        RunRecorder(tmp_path / "runs", model, "Q?")
    assert not (tmp_path / "runs").exists()


def test_run_start_records_identity_corpus_prices_and_config(tmp_path: Path) -> None:
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    (text_dir / "original.txt").write_bytes(b"[Page 1]\nHi\n")
    with RunRecorder(
        tmp_path / "runs",
        MODEL,
        "Q?",
        label="case-1",
        text_dir=text_dir,
        extra={"max_turns": 5},
    ) as rec:
        rec.end("answered", [])
    start = events(rec.path)[0]
    assert rec.path.name == f"{rec.run_id}.jsonl" and "-case-1-" in rec.run_id
    assert start["event"] == "run_start" and start["run_id"] == rec.run_id
    assert start["model"] == MODEL and start["question"] == "Q?"
    assert set(start["git"]) == {"commit", "dirty"}
    assert start["corpus"] == {
        "original.txt": hashlib.sha256(b"[Page 1]\nHi\n").hexdigest()
    }
    assert start["prices"]["cache_read"] == 0.10
    assert "platform.claude.com" in start["prices"]["source"]
    assert start["config"] == {"max_turns": 5}


def test_full_run_writes_requests_responses_and_tools_in_order(tmp_path: Path) -> None:
    tool_use = {
        "type": "tool_use",
        "id": "tu_1",
        "name": "grep",
        "input": {"pattern": "7.06"},
    }
    thinking = {"type": "thinking", "thinking": "", "signature": "sig"}
    messages: list[MessageParam] = [{"role": "user", "content": "Q?"}]
    params: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": 1024,
        "system": "S",
        "tools": [{"name": "grep"}],
        "messages": messages,
    }
    with RunRecorder(tmp_path, MODEL, "Q?", text_dir=tmp_path / "none") as rec:
        rec.model_request(1, params)
        first = response(stop_reason="tool_use", content=[thinking, tool_use])
        assert rec.model_call(1, first) == pytest.approx(USAGE_COST)
        messages.append({"role": "assistant", "content": first.content})
        rec.tool_call(
            1, "tu_1", "grep", {"pattern": "7.06"}, "original.txt:12: 7.06", False
        )
        rec.model_request(2, params | {"messages": messages})
        rec.model_call(2, response())
        rec.end("answered", messages)

    log = events(rec.path)
    assert [e["event"] for e in log] == [
        "run_start",
        "model_request",
        "model_call",
        "tool_call",
        "model_request",
        "model_call",
        "run_end",
    ]
    assert log[1]["params"] == {
        "model": MODEL,
        "max_tokens": 1024,
        "system": "S",
        "tools": [{"name": "grep"}],
        "message_count": 1,
    }
    assert log[4]["params"]["system"] == "<unchanged>"
    assert log[4]["params"]["tools"] == "<unchanged>"
    assert log[4]["params"]["message_count"] == 2
    assert [b["type"] for b in log[2]["response"]["content"]] == [
        "thinking",
        "tool_use",
    ]
    assert log[3]["result"] == "original.txt:12: 7.06" and log[3]["is_error"] is False
    end = log[-1]
    assert end["stop_reason"] == "answered" and end["turns"] == 2
    assert (
        end["cost_usd"] == pytest.approx(2 * USAGE_COST)
        and end["cost_complete"] is True
    )
    assert end["input_tokens"] == 2000 and end["cache_read_input_tokens"] == 20000
    assert end["messages"][1]["content"][1]["name"] == "grep"  # SDK blocks serialised


def test_same_label_same_second_gets_distinct_runs(tmp_path: Path) -> None:
    first = RunRecorder(tmp_path, MODEL, "Q?", label="ask", text_dir=tmp_path / "none")
    second = RunRecorder(tmp_path, MODEL, "Q?", label="ask", text_dir=tmp_path / "none")
    first.end("answered", [])
    second.end("answered", [])
    assert first.path != second.path and first.path.exists() and second.path.exists()


def test_events_are_on_disk_while_the_run_is_open(tmp_path: Path) -> None:
    rec = RunRecorder(tmp_path, MODEL, "Q?", text_dir=tmp_path / "none")
    rec.model_call(1, response())
    assert [e["event"] for e in events(rec.path)] == ["run_start", "model_call"]
    rec.end("answered", [])


def test_crash_after_a_request_marks_cost_incomplete(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="timed out"):
        with RunRecorder(tmp_path, MODEL, "Q?", text_dir=tmp_path / "none") as rec:
            rec.model_request(1, {"model": MODEL, "messages": []})
            raise RuntimeError("timed out")
    end = events(rec.path)[-1]
    assert end["event"] == "run_end" and end["stop_reason"] == "error"
    assert end["cost_complete"] is False and not rec.cost_complete
    assert "RuntimeError: timed out" in end["error"]
    assert "messages" not in end


def test_end_twice_writes_one_run_end(tmp_path: Path) -> None:
    with RunRecorder(tmp_path, MODEL, "Q?", text_dir=tmp_path / "none") as rec:
        rec.end("answered", [])
        rec.end("answered", [])
    assert [e["event"] for e in events(rec.path)] == ["run_start", "run_end"]
