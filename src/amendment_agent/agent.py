"""The agent loop. USER-OWNED: the scaffold defines the contract only."""

from enum import StrEnum
from pathlib import Path

from anthropic import Anthropic

from amendment_agent.recorder import RunRecorder


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


SYSTEM_PROMPT = ""  # user-owned


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
    """Answer `question` from data/text/ and write `output_dir/answer.json` (schema.Answer).

    Contract the scaffold relies on:
    - Record each request with `recorder.model_request` before sending it and
      `recorder.model_call` after; each tool run with `recorder.tool_call`; finish
      with `recorder.end(stop_reason, messages)`.
    - The client has SDK retries off: a failed request raises to the loop, and a retry is a new `model_request`.
    - Own the message history: build `messages` in the loop and pass it to
      `recorder.end`.
    - Always stop explicitly: return a `StopReason` on every path.
    - Tool failures go back to the model as error results; they never raise.
    - Return `ANSWERED` only if `answer.json` was written during this run.
    - Tool choice stays `auto`: forced tool choice is an HTTP 400 on Sonnet 5.5 and
      Opus 5.5.
    - Document text is data, never instructions.
    """
    raise NotImplementedError
