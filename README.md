# github-agent

LLM-powered tools for reviewing GitHub issues and pull requests with a
self-hosted model. Talks to any OpenAI-compatible endpoint (Ollama, LiteLLM, ...).

## Requirements

- [uv](https://docs.astral.sh/uv/)
- A running OpenAI-compatible LLM endpoint

## Usage

```bash
uv run issue-review.py digitalfabrik/lunes-cms 914
```

By default the review is printed to stdout (dry run). To post it as an issue
comment instead:

```bash
uv run issue-review.py digitalfabrik/lunes-cms 914 --post
```

Posting is idempotent: the review comment carries a hidden marker
(`<!-- llm-issue-review -->`) and is updated in place on subsequent runs.

### Options

| Flag | Effect |
|------|--------|
| `--post` | Post/update the review as an issue comment (requires `GITHUB_TOKEN`) |
| `--show-prompt` | Print the assembled prompt and exit without calling the LLM |
| `--model NAME` | Override the model for this run |

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_BASE_URL` | `http://localhost:11434` | OpenAI-compatible endpoint base URL |
| `LLM_MODEL` | `gemma4:31b` | Model name |
| `LLM_API_KEY` | *(empty)* | Bearer token for the LLM endpoint (e.g. LiteLLM) |
| `GITHUB_TOKEN` | *(empty)* | GitHub API token; optional for reading public repos, required for `--post` |

## What the review contains

The model receives the issue (body, comments, current labels) together with
the repository's full label set and produces:

- **Classification** — bug report, feature request, task, or question
- **Completeness** — whether the issue is actionable, what information is missing
- **Labels** — suggestions chosen only from the repository's existing labels
- **Open questions** — design/scoping questions to resolve before implementation
- **Suggested next step**

## Ideas

Rough roadmap, in order of expected value:

1. **Code grounding** — clone/checkout the target repo, grep for keywords from
   the issue, and feed matching file paths and snippets into the prompt. Turns
   "restate the issue" into "start at `services/audio_generation.py`, a
   migration is needed". The 128k context leaves plenty of room.
2. **Two-step prompting** — step 1 extracts structured JSON (`type`,
   `suggested_labels`, `search_keywords`, `open_questions`) via constrained
   output; step 2 uses it to drive code search and compose the final review.
   More reliable than one giant prompt for mid-size models.
3. **Duplicate detection** — embed all open issues once (e.g.
   `nomic-embed-text` via Ollama), cosine-match new issues against them.
   Cheap; no LLM call needed for retrieval.
4. **PR review** — port of the Forgejo `llm-pr-review.py` to the GitHub API:
   diff + commit messages in, idempotent review comment out.
5. **Eval harness** — run the agent on already-closed issues and compare
   suggested labels against actual labels and suggested files against the
   files touched by the fixing PR. Measures whether the model is good enough
   before trusting it.
6. **Delivery modes** — beyond the CLI: a cron sweep over unlabeled issues, or
   a GitHub Action on `issues: opened`.
