# github-agent

CLI tools that triage GitHub issues and pull requests with a self-hosted
LLM. Each tool is a standalone script talking to any OpenAI-compatible
endpoint (Ollama, LiteLLM, Open WebUI, ...):

| Script | What it does |
|--------|--------------|
| [`issue-review.py`](#issue-reviewpy) | Reviews an issue: classification, completeness, label suggestions, open questions, effort estimate, test coverage |
| [`pr-labels.py`](#pr-labelspy) | Suggests labels for a pull request from its description, commits and diff |
| [`issue-duplicates.py`](#issue-duplicatespy) | Finds likely duplicates of an issue among the open issues |

All tools are read-only by default: they print to stdout and touch nothing
on GitHub. Writing (posting a review comment, applying labels) is opt-in
via flag and requires a `GITHUB_TOKEN`.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (no install step — `uv run`
resolves dependencies on the fly) and access to an OpenAI-compatible LLM
endpoint, configured via environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_BASE_URL` | `http://localhost:11434` | OpenAI-compatible endpoint base URL |
| `LLM_MODEL` | `gemma4:31b` | Model name |
| `LLM_API_KEY` | *(empty)* | Bearer token for the LLM endpoint (e.g. Open WebUI, LiteLLM) |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model for duplicate detection |
| `GITHUB_TOKEN` | *(empty)* | GitHub API token; optional for reading public repos, required for writing |

With [direnv](https://direnv.net/) installed, put the environment into
`.envrc` (gitignored, holds your API key) and run `direnv allow`:

```bash
export LLM_BASE_URL=https://<your-openwebui-host>/api
export LLM_API_KEY=sk-...
export LLM_MODEL=verdigado-think
```

Note for Open WebUI: the base URL must end in `/api` — its
OpenAI-compatible routes live under `/api/v1/...` and all require a Bearer
API key (Settings → Account → API Keys).

## issue-review.py

Reviews a single issue. The model receives the issue (body, comments,
current labels) together with the repository's full label set and produces
a triage review:

- **Classification** — bug report, feature request, task, or question
- **Completeness** — whether the issue is actionable, what information is missing
- **Labels** — suggestions chosen only from the repository's existing labels
- **Open questions** — design/scoping questions to resolve before implementation
- **Effort estimate** — rough S/M/L sizing, only when the issue is actionable
- **Test coverage** — whether e2e tests make sense, in the implementing PR or
  as a follow-up issue
- **Suggested next step**

```bash
uv run issue-review.py digitalfabrik/lunes-cms 953
```

| Flag | Effect |
|------|--------|
| `--post` | Post/update the review as an issue comment (requires `GITHUB_TOKEN`) |
| `--show-prompt` | Print the assembled prompt and exit without calling the LLM |
| `--model NAME` | Override the model for this run |

Posting is idempotent: the review comment carries a hidden marker
(`<!-- llm-issue-review -->`) and is updated in place on subsequent runs.

### Example

```text
$ uv run issue-review.py digitalfabrik/lunes-cms 953
Fetching issue digitalfabrik/lunes-cms#953 ...
Issue: "Invalid csv structure is not rejected during import" — 0 comment(s), 23 repo label(s)

### Classification
Bug report. The CSV import process fails to validate the required file
structure, allowing incomplete data to be imported instead of rejecting it.

### Completeness
Actionable. The issue provides reproduction steps and a video demonstration
of the failure.

### Labels
- Add `python`: The import validation logic resides in the backend.
- Add `prio: medium`: Data integrity issues during import are typically
  medium priority.
- Keep `ready`: The bug is clearly described and reproducible.

### Open questions
Does the system need to provide a specific error message identifying which
columns are missing, or is a generic "invalid structure" error sufficient?

### Effort estimate
M. Requires implementing a schema validation check before the import loop
and ensuring the process halts without creating partial records.

### Test coverage
Yes. The implementing PR should include a test case specifically using a
malformed CSV to verify that the import is rejected.

### Suggested next step
Ready to implement.
```

## pr-labels.py

Fetches a pull request (description, commit messages, diff) and asks the
model which of the repository's labels to add or remove.

```bash
uv run pr-labels.py digitalfabrik/lunes-cms 958
```

| Flag | Effect |
|------|--------|
| `--apply` | Apply suggested additions to the PR (requires `GITHUB_TOKEN`) |
| `--show-prompt` | Print the assembled prompt and exit without calling the LLM |
| `--model NAME` | Override the model for this run |

Label removals are only ever suggested, never applied — `--apply` adds
labels but leaves existing ones untouched.

### Example

```text
$ uv run pr-labels.py digitalfabrik/lunes-cms 958
Fetching PR digitalfabrik/lunes-cms#958 ...
PR: "Hide 'Add another' link for alternative words on the change page" — 1 commit(s), 23 repo label(s)

+ python: The pull request modifies Python code in the admin and test files.
+ ui-ux: The change improves the user interface by removing a confusing link
  on the word change page.
```

## issue-duplicates.py

Embeds all open issues of the repository via the `/v1/embeddings` endpoint
and ranks them by cosine similarity against the given issue. Embeddings are
cached in `.cache/` and only re-computed for new or changed issues.

```bash
uv run issue-duplicates.py digitalfabrik/lunes-cms 914
```

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
   diff + commit messages in, idempotent review comment out. The label
   suggestion part is done, see `pr-labels.py`.
5. **Eval harness** — run the agent on already-closed issues and compare
   suggested labels against actual labels and suggested files against the
   files touched by the fixing PR. Measures whether the model is good enough
   before trusting it.
6. **Delivery modes** — beyond the CLI: a cron sweep over unlabeled issues, or
   a GitHub Action on `issues: opened`.
