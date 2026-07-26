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

### Example

```text
$ uv run issue-review.py digitalfabrik/lunes-cms 914
Fetching issue digitalfabrik/lunes-cms#914 ...
Issue: "Add a pronunciation field" — 0 comment(s), 22 repo label(s)

### Classification
Feature request. The issue asks for a new pronunciation field to override
default AI audio generation for loan words.

### Completeness
Insufficient. While the goal is clear, it lacks:
- The specific entity or model where this field should be added
  (e.g., Vocabulary/Word entry).
- Acceptance criteria for how the substitution logic should handle multiple
  occurrences of a word within a sentence.
- UI design specifications for the suggested "hint" text.

### Labels
- Remove `ready`: The implementation details for sentence replacement are too
  vague to be considered ready for development.
- Add `ui-ux`: A new input field and helper text are required in the interface.
- Add `python`: Changes to the audio generation backend logic are necessary.
- Add `needs discussion`: The technical approach to replacing words within
  sentences needs to be defined.

### Open questions
- Which database model/endpoint should store this pronunciation value?
- Should the replacement in sentences be a simple string replace or handle
  case sensitivity and stemming?
- Does this field apply globally to a word across all sentences, or is it
  per-sentence?

### Suggested next step
Ask the reporter to specify which object/model needs the new field and how
the sentence substitution should behave
```

### Options

| Flag | Effect |
|------|--------|
| `--post` | Post/update the review as an issue comment (requires `GITHUB_TOKEN`) |
| `--show-prompt` | Print the assembled prompt and exit without calling the LLM |
| `--model NAME` | Override the model for this run |

## Duplicate detection

```bash
uv run issue-duplicates.py digitalfabrik/lunes-cms 914
```

Embeds all open issues of the repository via the `/v1/embeddings` endpoint and
ranks them by cosine similarity against the given issue. Embeddings are cached
in `.cache/` and only re-computed for new or changed issues.

| Flag | Effect |
|------|--------|
| `--top N` | Number of matches to show (default: 5) |
| `--min-score X` | Minimum similarity to report (default: 0.5) |
| `--no-cache` | Ignore and rebuild the embedding cache |
| `--llm` | Rank candidates with the chat model instead of embeddings |

If the endpoint offers no embedding model (e.g. an Open WebUI backed only by
a LiteLLM chat proxy), use `--llm`: the chat model gets the target issue plus
all candidate titles/excerpts and returns scored matches as JSON. Slower and
less precise than embeddings, but needs nothing beyond the chat model.

### Local configuration

With [direnv](https://direnv.net/) installed, put the environment into
`.envrc` (gitignored, holds your API key):

```bash
export LLM_BASE_URL=https://<your-openwebui-host>/api
export LLM_API_KEY=sk-...
export LLM_MODEL=verdigado-think
```

Then run `direnv allow`. Note for Open WebUI: the base URL must end in
`/api` — its OpenAI-compatible routes live under `/api/v1/...` and all
require a Bearer API key (Settings → Account → API Keys).

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_BASE_URL` | `http://localhost:11434` | OpenAI-compatible endpoint base URL |
| `LLM_MODEL` | `gemma4:31b` | Model name |
| `LLM_API_KEY` | *(empty)* | Bearer token for the LLM endpoint (e.g. Open WebUI, LiteLLM) |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model for duplicate detection |
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
3. ~~**Duplicate detection**~~ — done, see `issue-duplicates.py`.
4. **PR review** — port of the Forgejo `llm-pr-review.py` to the GitHub API:
   diff + commit messages in, idempotent review comment out.
5. **Eval harness** — run the agent on already-closed issues and compare
   suggested labels against actual labels and suggested files against the
   files touched by the fixing PR. Measures whether the model is good enough
   before trusting it.
6. **Delivery modes** — beyond the CLI: a cron sweep over unlabeled issues, or
   a GitHub Action on `issues: opened`.
