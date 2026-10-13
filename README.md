# amendment-agent

An AI agent that answers questions about one credit agreement and its amendments
(Envista Holdings, 2019, from SEC EDGAR). For each question, it finds the version of
the rule that applied to the period asked about and cites the exact lines it used.

## How it works

The agent decides its own steps. It has four tools over a folder of plain-text
filings: list the files, search them, read lines, and write the answer. It can only
read the filings and only write to its own output folder.

| id | Document | Dated |
|---|---|---|
| `original` | Credit Agreement | 2019-09-20 |
| `amendment-1` | Amendment No. 1: restates the agreement, with covenant relief | 2020-05-06 |
| `amendment-2` | Amendment No. 2: §7.03(l) | 2020-05-19 |
| `amendment-3` | Amendment No. 3: "Applicable Rate"; restates §7.06(a)–(b) | 2021-02-09 |

The hard part is picking the right version. A later amendment usually replaces earlier
text, but only from its effective date, unless it says it reaches back. So the
latest document isn't always the answer.

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

The default model is `claude-sonnet-5-5`; use `--model claude-opus-5-5` for Opus.

## The answer

Each run writes `outputs/<run_id>/answer.json` and a full transcript in
`runs/<run_id>.jsonl`:

```json
{
  "status": "answered",
  "value": "not tested",
  "explanation": "Amendment No. 1 restated Section 7.06(a) to test the maximum Consolidated Leverage Ratio (3.75:1) only for the quarter ending about March 31, 2020 and for quarters ending on or after about June 30, 2021. [...]",
  "citations": [
    {"document": "amendment-1", "section": "7.06(a)", "lines": [1363, 1363], "quote": "(i) as of the end of the fiscal quarter [...]"}
  ]
}
```

If the documents don't answer the question, the status is `insufficient_evidence`
and there is no value. Before saving, the agent's citations are checked: the quote
must appear in the cited lines, and the section must match.

## When a run stops

A run stops when the answer is written, or at one of these limits:

- **Turns:** a fixed number of model calls.
- **Cost:** checked before each call, so a run can go over by up to one call.
- **Repetition:** an identical tool call isn't run again (the model is pointed to the
  earlier result); a third one ends the run.
- **Errors:** temporary API errors (network, rate limits, server errors) are retried 3
  times; other API errors end the run.

Each ending is recorded with its own reason (`answered`, `no_answer`, `max_turns`,
`max_cost`, `repeated_call`, `refusal`, `max_tokens`, `provider_error`).

## Results

8 test questions, each checked by hand against the documents, in
[`evals/cases.jsonl`](evals/cases.jsonl). Each ran 3 times per model:

| Model | Right answer | Right citations | Cost per question |
|---|---|---|---|
| Sonnet 5.5 (default) | 24/24 | 24/24 | $0.03 |
| Opus 5.5 | 24/24 | 23/24 | $0.10 |

Opus was no more accurate and cost about 3 times as much. Its one miss cited the right
lines under a broader label (`§7.06` instead of `§7.06(a)`).

**What the scoring checks:** the answer matches the expected one (allowing for
formats like `3.75:1` vs `3.75 to 1.00`), and the expected sections are cited.
Extra citations are flagged, not failed. **What it doesn't check:** that the cited
lines support the answer, or that the explanation is right. A second AI model reviewed
the answers for that; it agreed with every score and found one badly worded test
question, now fixed.

How the scores got here:

- The first runs scored 4/8 (Sonnet) and 7/8 (Opus) on citations, but every miss was a problem in the scoring
  or the data, not the agent: the scorer was too strict about section labels, and one
  SEC filing is missing a section number. Both were fixed.
- A code review by a second AI model found six bugs; all are fixed and tested.
- Once, Sonnet cited only a later amendment that doesn't apply to the period. It
  happened in 1 of 8 runs of that question.

Results are only compared when the questions, scoring code and documents are the same;
each results file in `evals/results/` records all three. Current results: commit
`0d8dffd`, `evals/results/2026-10-13T01-*.json`.

## Example runs

Both for the question "What maximum Consolidated Leverage Ratio applies to the quarter
ending September 30, 2020?":

- **Success:** [transcript](runs/2026-10-12T20-21-38Z-ask-e4248a.jsonl),
  [answer](outputs/2026-10-12T20-21-38Z-ask-e4248a/answer.json). Answers "not tested"
  from Amendment No. 1, in 5 steps for $0.06. Its first answer was rejected for a
  wrong citation, and it fixed it.
- **Failure:** [transcript](runs/2026-10-12T18-51-25Z-ask-7f6c20.jsonl),
  [answer](outputs/2026-10-12T18-51-25Z-ask-7f6c20/answer.json). Right conclusion,
  but it cited Amendment No. 3, which only applies from 2021, and wrote a sentence
  instead of a short value. **Fixed by** telling the agent which version applies to a
  past period, and that the value must be the bare answer.

## Limitations

- Answers describe what these documents say for a period, not the agreement today.
- A small test set: enough to catch the problems above, not to measure error rates.
- Section detection fits this agreement's formatting; other companies' filings need
  more work.
- Filing defects are fixed one at a time, as they're found.
- Next: measure one agent change against these results.

## Layout

```
src/amendment_agent/   agent, system_prompt.md, tools, citations, graders (core)
                       cli, settings, manifest, edgar, extract, recorder, schema, evals
data/manifest.yaml     the filings (downloaded and extracted text isn't committed)
evals/                 test questions (cases.jsonl) and results
runs/, outputs/        transcripts and answers (only the examples are committed)
```

## Development

`make check` runs the linters, type checks and offline tests. CI runs the same on every
push and pull request.
