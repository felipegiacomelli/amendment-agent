"""The agent loop. USER-OWNED: the scaffold defines the contract only."""

import json
import time
from collections import Counter
from enum import StrEnum
from pathlib import Path

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    InternalServerError,
    RateLimitError,
)
from anthropic.types import Message, MessageParam, ToolResultBlockParam
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

from amendment_agent.recorder import RunRecorder
from amendment_agent.schema import ANSWER_FILENAME, load_answer
from amendment_agent.tools import TOOL_SCHEMAS, ToolResult, execute_tool

MAX_RETRIES = 3
MAX_REPEATS = 3
RETRYABLE = (
    APIConnectionError,
    RateLimitError,
    InternalServerError,
)


class StopReason(StrEnum):
    """Why a run ended. A starting set; change it to fit your policies."""

    ANSWERED = "answered"
    NO_ANSWER = "no_answer"
    MAX_TURNS = "max_turns"
    MAX_COST = "max_cost"
    REPEATED_CALL = "repeated_call"
    REFUSAL = "refusal"
    MAX_TOKENS = "max_tokens"
    PROVIDER_ERROR = "provider_error"


SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.md")


def run_agent(
    question: str,
    *,
    client: Anthropic,
    model: str,
    recorder: RunRecorder,
    output_dir: Path,
    max_turns: int,
    max_cost_usd: float,
) -> StopReason:
    """Answer `question` from data/text/ and write `output_dir/answer.json`.

    Contract the scaffold relies on:
    - Record each request with `recorder.model_request` before sending it and
      `recorder.model_call` after; each tool run with `recorder.tool_call`; finish
      with `recorder.end(stop_reason, messages)`.
    - The client has SDK retries off: a failed request raises to the loop, and a retry
      is a new `model_request`.
    - Own the message history: build `messages` in the loop and pass it to
      `recorder.end`.
    - Always stop explicitly: return a `StopReason` on every path.
    - Tool failures go back to the model as error results; they never raise.
    - Return `ANSWERED` only if `answer.json` was written during this run.
    - Tool choice stays `auto`: forced tool choice is an HTTP 400 on Sonnet 5.5 and
      Opus 5.5.
    - Document text is data, never instructions.
    """
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    messages: list[MessageParam] = [{"role": "user", "content": question}]
    params: MessageCreateParamsNonStreaming = {
        "model": model,
        "max_tokens": 8192,
        "cache_control": {"type": "ephemeral"},
        "system": system_prompt,
        "tools": TOOL_SCHEMAS,
        "output_config": {"effort": "medium"},
        "messages": messages,
    }

    turn = 0
    cost = 0.0
    seen: Counter[tuple[str, str]] = Counter()
    wrote_answer = False  # a file left in output_dir by an earlier run doesn't count

    while turn < max_turns and cost < max_cost_usd:
        params["messages"] = messages

        response = _send(client, params, recorder, turn)

        if response is None:
            stop_reason = StopReason.PROVIDER_ERROR
            break

        cost += recorder.model_call(turn, response)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            stop_reason = StopReason.REFUSAL
            break
        elif response.stop_reason == "max_tokens":
            stop_reason = StopReason.MAX_TOKENS
            break

        tool_results: list[ToolResultBlockParam] = []
        for block in response.content:
            if block.type == "tool_use":
                key = (block.name, json.dumps(block.input, sort_keys=True))
                seen[key] += 1
                if seen[key] > 1:
                    result = ToolResult(
                        text=(
                            "error: you already made this exact call, "
                            "its result is above. "
                            "Use it, or try a different call."
                        ),
                        is_error=True,
                    )
                else:
                    result = execute_tool(
                        block.name, block.input, output_dir=output_dir
                    )
                    if block.name == "write_answer" and not result.is_error:
                        wrote_answer = True

                recorder.tool_call(
                    turn=turn,
                    tool_use_id=block.id,
                    name=block.name,
                    tool_input=block.input,
                    result=result.text,
                    is_error=result.is_error,
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result.text,
                        "is_error": result.is_error,
                    }
                )

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        if max(seen.values(), default=0) >= MAX_REPEATS:
            stop_reason = StopReason.REPEATED_CALL
            break

        if (
            response.stop_reason
            in ("pause_turn", "stop_sequence", "model_context_window_exceeded")
            or response.stop_reason is None
        ):
            stop_reason = StopReason.PROVIDER_ERROR
            break
        elif response.stop_reason == "end_turn":
            stop_reason = StopReason.NO_ANSWER
            break

        turn += 1
    else:
        # The loop's own condition ended it: say which limit. Every other ending breaks.
        if cost >= max_cost_usd:
            stop_reason = StopReason.MAX_COST
        else:
            stop_reason = StopReason.MAX_TURNS

    # A valid answer.json written in this run overrides whatever ended the loop
    if wrote_answer:
        try:
            load_answer(output_dir / ANSWER_FILENAME)
            stop_reason = StopReason.ANSWERED
        except ValueError:
            pass

    recorder.end(stop_reason, messages)

    return stop_reason


def _send(
    client: Anthropic,
    params: MessageCreateParamsNonStreaming,
    recorder: RunRecorder,
    turn: int,
) -> Message | None:
    """Request with retries, None when it can't get a response."""
    for attempt in range(MAX_RETRIES + 1):
        recorder.model_request(turn, dict(params))
        try:
            return client.messages.create(**params)
        except RETRYABLE as error:
            if attempt == MAX_RETRIES:
                return None
            time.sleep(_retry_delay(error, attempt))
        except APIStatusError:
            return None
    return None


def _retry_delay(error: Exception, attempt: int, multiplier: int = 2) -> float:
    """Server's retry-after if it sent one."""
    if isinstance(error, APIStatusError):
        try:
            return float(error.response.headers.get("retry-after", ""))
        except ValueError:
            pass
    return float(multiplier**attempt)
