# amendment-agent

An autonomous agent that answers questions about one credit agreement family: Envista
Holdings' 2019 credit agreement and its three amendments, from SEC EDGAR. For each
question it finds the provision that applies to the period asked about, within this
document set, and cites the document, section and line range.

## How it works

The agent decides its own steps, the way a coding agent works through a repository.
It lists the documents, greps for terms, reads line ranges and writes an answer file.
It has four tools (`list_files`, `grep`, `read_file`, `write_output`) over a folder of
plain-text filings, and no search index. Reads are confined to the filings and writes
to the run's own output folder. A run ends when the answer is written, or at a turn,
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
| Agent loop, tools, graders | in progress, hand-written |

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
  "explanation": "Amendment No. 1 and Amendment No. 3 both leave the quarter ending on or about September 30, 2020 outside the leverage test.",
  "citations": [
    {"document": "amendment-1", "section": "7.06(a)", "lines": [1363, 1363]},
    {"document": "amendment-3", "section": "7.06(a)", "lines": [22, 22]}
  ]
}
```

`status` can also be `insufficient_evidence`. The transcript for the same run is
`runs/<run_id>.jsonl`.

## Evidence

- Successful run (transcript and answer): *not yet produced*
- Failed run and analysis: *not yet produced*
- Before/after comparison on a fixed case set: *not yet produced*

## Limitations

- A historical document set. Answers state what these documents provide for a period,
  not the agreement's status today.
- Citation grading checks that cited sections and line ranges exist and match the
  expected ones. It does not check that the cited text supports the answer.
- Cost-limit semantics: *to be documented with the agent loop*.
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
