# amendment-agent

An autonomous agent that answers questions about one credit agreement family: Envista
Holdings' 2019 credit agreement and its three amendments, from SEC EDGAR. For each
question it finds the provision that applies to the period asked about, within this
document set, and cites the document, section and line range.

## How it works

The agent is designed to decide its own steps, the way a coding agent works through a repository.
It lists the documents, greps for terms, reads line ranges and writes an answer file.
It has four tools (`list_files`, `grep`, `read_file`, `write_answer`) over a folder of
plain-text filings, and no search index. Reads are confined to the filings and writes
to the run's own output folder. A run is meant to end when the answer is written, or at a turn,
cost or repetition limit.

| id | Document | Dated |
|---|---|---|
| `original` | Credit Agreement | 2019-09-20 |
| `amendment-1` | Amendment No. 1: restates the agreement, with covenant relief | 2020-05-06 |
| `amendment-2` | Amendment No. 2: §7.03(l) | 2020-05-19 |
| `amendment-3` | Amendment No. 3: "Applicable Rate"; restates §7.06(a)–(b) | 2021-02-09 |

Later amendments override earlier text, so the right answer is often not in the
original agreement. Document dates set the order; the question's period picks the
version; conditions in the text can still change what applies.

## Status

| Part | State |
|---|---|
| EDGAR fetcher (rate-limited), text extraction, document index | implemented, tested offline |
| Run recorder: JSONL transcript with config, git revision, corpus hashes, every request, response, tool call, usage and cost | implemented, tested offline |
| Answer and eval-case schema; eval runner with fail-closed grading and comparable results files | implemented, tested offline |
| Agent loop and tools | hand-written, tested offline with a scripted fake client |
| Graders | in progress, hand-written |

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
make install
cp .env.example .env        # set ANTHROPIC_API_KEY and SEC_USER_AGENT ("Name email")
uv run amendment-agent fetch
uv run amendment-agent extract
uv run amendment-agent ask "What maximum Consolidated Leverage Ratio applies to the quarter ending September 30, 2020?"
uv run amendment-agent eval
```

The default model is `claude-sonnet-5-5`. `claude-opus-5-5` is also supported, via
`AGENT_MODEL` or `--model`.

## Answer format

Each run writes `outputs/<run_id>/answer.json`:

```json
{
  "status": "answered",
  "value": "not tested",
  "explanation": "Amendment No. 1 restated Section 7.06(a) to test the maximum Consolidated Leverage Ratio (3.75:1) only for the quarter ending about March 31, 2020 and for quarters ending on or after about June 30, 2021. [...]",
  "citations": [
    {
      "document": "amendment-1",
      "section": "7.06(a)",
      "lines": [1363, 1363],
      "quote": "(i) as of the end of the fiscal quarter of the Company ending on or about March 31, 2020 to be greater than 3.75:1, and (ii) [...]"
    }
  ]
}
```

`value` is the bare answer, since it is graded on its own; everything else goes in
`explanation`, which is not graded. `status` can also be `insufficient_evidence`, with
no value and no required citations; the searches it made are in the transcript.
`write_answer` checks every citation before writing: the document exists, the range
is at most 20 lines inside one section, the quote is in those lines, and the section
is the heading above them or one the lines name. The transcript for the same run is
`runs/<run_id>.jsonl`.

## Run limits and endings

Every run ends with one stop reason, recorded in the transcript's `run_end`:

| Stop reason | When |
|---|---|
| `answered` | a valid `answer.json` was written during the run, whatever ended the loop |
| `no_answer` | the model ended its turn without a valid answer file |
| `max_turns` | the turn limit was reached: exactly that many model calls were made |
| `max_cost` | the cost limit was reached (see below) |
| `repeated_call` | the same tool call (same tool, same input) was made a third time |
| `refusal`, `max_tokens` | the API's stop reason of the same name |
| `provider_error` | the API failed after retries, rejected the request, or stopped for a reason this agent doesn't handle (`pause_turn`, `stop_sequence`, `model_context_window_exceeded`) |

- **Cost limit: a threshold, not a guarantee.** Before each model call the loop checks
  the total cost so far; once it is at or over the limit, no further call is sent. A
  run can therefore exceed the limit by up to one call: roughly the whole history as
  input plus `max_tokens` of output.
- **Repetition.** A call identical to an earlier one in the run is not executed: the
  model gets an error pointing it to the earlier result, which is still in its
  history. The third identical call ends the run. A revisit after other calls (read A,
  read B, read A) is refused the same way, but the run goes on.
- **Retries.** Connection errors, rate limits and 5xx are retried up to 3 times, with
  the server's `retry-after` or 1, 2, 4 s backoff; each attempt is its own request in
  the transcript. Other API errors end the run at once. API errors are never shown to
  the model as tool errors.

## Evidence

All on `claude-sonnet-5-5`, question: "What maximum Consolidated Leverage Ratio applies
to the fiscal quarter ending on or about September 30, 2020?" (eval case
`leverage-q3-2020`).

- **Successful run:** [`runs/2026-10-12T20-21-38Z-ask-e4248a.jsonl`](runs/2026-10-12T20-21-38Z-ask-e4248a.jsonl),
  answer [`outputs/2026-10-12T20-21-38Z-ask-e4248a/answer.json`](outputs/2026-10-12T20-21-38Z-ask-e4248a/answer.json).
  Commit `32e79de`, clean tree; 5 turns, $0.056. It answers `not tested` and cites
  Amendment No. 1's §7.06(a), the version in effect for that quarter. Its first
  `write_answer` was rejected (an Amendment No. 3 citation under the wrong section);
  the model read the error and fixed the citation on the next call.
- **Failed run:** [`runs/2026-10-12T18-51-25Z-ask-7f6c20.jsonl`](runs/2026-10-12T18-51-25Z-ask-7f6c20.jsonl),
  answer [`outputs/2026-10-12T18-51-25Z-ask-7f6c20/answer.json`](outputs/2026-10-12T18-51-25Z-ask-7f6c20/answer.json).
  Commit `ef5d746` with uncommitted changes. It reached the right conclusion but
  failed in two ways:
  - It cited only Amendment No. 3's restated §7.06(a). That version applies from the
    quarter ending on or about March 31, 2021 and doesn't reach back, so it doesn't
    govern the 2020 quarter; Amendment No. 1's §7.06(a) does. The prompt had no rule
    for which version applies to an earlier period, so the agent took the latest
    text. **Change:** the system prompt now states the precedence rule: an amendment
    applies from its effective date and changes an earlier period only where its text
    expressly covers that period (`d53342d`).
  - Its value was a sentence ("None – no maximum Consolidated Leverage Ratio applies
    (not tested) for the quarter ending September 30, 2020"), which no deterministic
    grader can match. **Change:** the prompt now says the value is graded on its own
    and holds only the answer (`b4c6e87`).
- Before/after comparison on a fixed case set: *not yet produced* (needs the graders).

## Limitations

- A historical document set. Answers state what these documents provide for a period,
  not the agreement's status today.
- Citation grading checks that cited sections and line ranges exist and match the
  expected ones. It does not check that the cited text supports the answer.
- The cost limit is checked between calls, so a run can exceed it by one call (see
  [Run limits and endings](#run-limits-and-endings)).
- Basic text extraction: page numbers count rendered pages rather than printed
  footers, and there is no structure-aware parsing.
- One agreement family; models limited to those with exact pricing in the recorder.

## Layout

```
src/amendment_agent/   cli, settings, manifest, edgar, extract, recorder, schema, evals
                       agent, tools, graders (hand-written core)
data/manifest.yaml     the filings; raw and extracted text are not committed
evals/cases.jsonl      eval cases; results in evals/results/
runs/, outputs/        per-run transcripts and answers (not committed by default)
```

## Development

`make check` runs lint, `mypy --strict` and the offline tests with a coverage floor.
CI runs the same on every push and pull request.
