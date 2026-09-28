#!/usr/bin/env python3
"""Review a GitHub PR with a self-hosted LLM.

Fetches the PR diff, the commit messages and the labels currently set, asks the
model for a review, and prints it (default) or posts it as an
idempotent marker comment (--post).
Provide an alternative system prompt over stdin when specifying --stdin-prompt.

Usage:
  python3 pr-review.py digitalfabrik/integreat-cms 914
  python3 pr-review.py digitalfabrik/integreat-cms 914 --post
  python3 pr-review.py digitalfabrik/integreat-cms 914 --prompt-path ai/PRMPT.md
  cat system-prompt.md | python3 pr-review.py \
    digitalfabrik/integreat-cms 914 --post --stdin-prompt

Environment:
  LLM_BASE_URL  OpenAI-compatible endpoint base (default http://localhost:11434)
  LLM_API_KEY   Bearer token for the LLM endpoint (optional, e.g. for LiteLLM)
  LLM_MODEL     Model name (default gemma4:31b)
  GITHUB_TOKEN  GitHub API token (optional for reading public repos,
                required for --post)
"""

import argparse
import os
import re
import sys

import requests

GITHUB_API = 'https://api.github.com'
LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'http://localhost:11434')
LLM_MODEL = os.environ.get('LLM_MODEL', 'gemma4:31b')
LLM_API_KEY = os.environ.get('LLM_API_KEY', '')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')

GITHUB_TIMEOUT = 30
LLM_TIMEOUT = 600
COMMENT_MARKER = '<!-- llm-pr-review -->'
PROMPT_PATH = 'REVIEW_PROMPT.md'

# Maximum diff size to send to the LLM (200 KB). Larger diffs are truncated.
MAX_DIFF_BYTES = 200_000

# Maximum diff size for a single file (20 KB). Larger per-file diffs (e.g.
# regenerated fixtures or lock files) are replaced by a placeholder so they
# can't push the interesting files of the PR past MAX_DIFF_BYTES.
MAX_FILE_DIFF_BYTES = 20_000

SYSTEM_PROMPT = """\
You are an experienced software engineer reviewing a GitHub Pull Request.

Analyse the diff, the commit messages and the PR's current labels and
write a concise review, reporting on:
1. Correctness (shown exemplary for Django, adapt accordingly for any other used
   framework):
 - Any model field change (new field, altered verbose_name/on_delete/
   null/blank) must be accompanied by a matching migration; the PR should
   not rely on `makemigrations` being run later. New migration files should
   have a short docstring on the `Migration` class describing intent (existing
   convention e.g. in `lunes_cms/*/migrations/` as well as `integreat_cms/`).
 - `on_delete` choices on ForeignKeys matter: flag `CASCADE` on
   relationships to user-generated content (jobs/units/words, feedback,
   etc.) where deleting the referenced row would silently wipe out
   unrelated content — prefer `SET_NULL`/`PROTECT` unless cascading
   deletion is clearly intentional.
2. Internationalization: New or changed `gettext`/`gettext_lazy` strings
 must be reflected in the corresponding file (in Django strings in `_(...)`
 and the translation file ending in `.po`) with no `#, fuzzy` markers and no
 empty `msgstr`. If a PR adds/changes translatable strings but doesn't touch
 the translation file, flag it.
3. Type safety: Unless apparent that the project does not use type hints for
 languages that are not already strongly typed (like python which is usually
 type-annotated by mypy + django-stubs), flag new functions/methods missing
 type hints. In the case of django and `request.user`, it is typed
 `User | AnonymousUser` by django-stubs — a bare assignment to a `User`-typed
 field needs either a real narrowing check or a `# type: ignore[...]` with a
 one-line comment explaining why the request is guaranteed authenticated
 (existing convention), not a silent or unexplained ignore.
4. Code quality gates: This repo might enforce conventions and formatting
 through tools like black, pylint, ruff or eslint. For any obviously used, flag
 obvious formatting drift or unjustified comments like `# pylint: disable=...`.
5. Security: Mechanisms like file upload validators must not be bypassed or
 weakened. Question any new `@csrf_exempt` view. Watch for OpenAI API key or
 other secrets leaking into logs, error messages, or committed example configs.
 The code for features like a CSV import handles user-supplied column data —
 check for unsafe assumptions about column contents.
6. Testing: New behavior (admin actions, CSV import, API endpoints,
 services) should come with tests under `tests/`. If a PR changes a
 shared function's signature, check that all call sites (including
 tests) were updated, not just the primary caller.
7. Obvious typos in code, comments, file paths, identifiers, and
 documentation. Only flag clear spelling mistakes — do not nitpick
 stylistic word choices.
8. Commit message:
 - The commit message must be generally useful: it should clearly
   describe what changed and, in the body, why. Flag messages that are
   vague ("fix stuff", "update"), tautological ("change X to X"), or that
   don't explain a non-obvious change.
 - Additional context (why the change is necessary) goes after a blank
   line in the body.
 - The commit messages should be consistent in style, e.g. if a commit starts
   its first line with an issue number, a colon and a space, all others should
   follow that style as well.
 Dependabot commits are exempt from these rules.

The message starts with the complete list of files changed in this PR.
The diff itself may be incomplete: oversized per-file diffs are replaced by a
"[diff omitted …]" placeholder and the overall diff may be truncated
("[diff truncated …]"). Whether a file is part of the PR must therefore only
ever be judged by the file list, never by the (possibly incomplete) diff. Never
claim that a file, migration, translation or test is missing from the PR when
the file list contains it — if its diff was omitted or truncated, instead begin
the review with a line that part of the diff could not be reviewed in full. Make
it a dropdown that contains more details about what was omitted, e.g.:

<details><summary><b><i>The diff of this PR was too large and could only partially be reviewed. Some content and context might be missing and relations and conclusions might be wrong.</i></b></summary>

- `integreat_cms/core/conf.py` only partially found in diff, interrupted by `[omitted]` marker
- `tests/b.py` only found in file list

</details>

Note how lines containing markdown require a blank line separating them from
lines with HTML tags in order to be interpreted by GitHub correctly. This is
why the text in `<summary>` is not using markdown but HTML tags.

If the diff is not truncated, just start the review with an italic line stating:

*Analyzed diff, but not the whole repository. Important context might be missing.*

Be specific and reference file paths and line numbers where possible.
Be concise. Do not approve or reject — provide comments only.
Do not mention things are "good" or "right" if there is nothing more to say about it.
Do not refer to yourself in the first person, if necessary always speak of
"the PR analysis" or "the review".

Rules:
- Base everything strictly on the PR text; do not invent details.
- Be specific and terse. No praise, no filler, no restating the whole PR.
- Do not propose implementation code.
"""


def die(message):
    print(f'Error: {message}', file=sys.stderr)
    sys.exit(1)


def log(message):
    print(message, file=sys.stderr)


def gh_headers(accept='application/vnd.github+json'):
    headers = {'Accept': accept}
    if GITHUB_TOKEN:
        headers['Authorization'] = f'Bearer {GITHUB_TOKEN}'
    return headers


def gh_get(url, params=None, accept='application/vnd.github+json'):
    response = requests.get(url, headers=gh_headers(accept), params=params,
                            timeout=GITHUB_TIMEOUT)
    if response.status_code != 200:
        die(f'GitHub API {url} returned {response.status_code}: '
            f'{response.text[:300]}')
    return response


def gh_get_paginated(url):
    results = []
    page = 1
    while True:
        batch = gh_get(url, params={'per_page': 100, 'page': page})
        results.extend(batch)
        if len(batch) < 100:
            return results
        page += 1


def fetch_pr(owner, repo, number):
    return gh_get(f'{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}')


def fetch_diff(owner, repo, number):
    url = f'{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}'
    response = gh_get(url, accept='application/vnd.github.v3.diff')
    return response.text.strip()


def fetch_prompt(owner, repo, path):
    url = f'https://raw.githubusercontent.com/{owner}/{repo}/HEAD/{path}'
    response = gh_get(url, accept='text/plain')
    return response.text.strip()


def fetch_comments(owner, repo, number):
    return gh_get_paginated(
        f'{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}/comments')


def fetch_commits(owner, repo, number):
    return gh_get_paginated(
        f'{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}/commits')


def path_from_diff_header(header_line):
    """
    Extracts the file path from a "diff --git a/... b/..." header line,
    e.g. 'diff --git a/setup.py b/setup.py' -> "setup.py".
    """
    return re.split(r' "?b/', header_line, maxsplit=1)[-1].rstrip('"')
    #return header_line.split(" b/", 1)[-1]


def split_diff_by_file(diff_text):
    """
    Splits a unified diff into a list of per-file chunks, each starting
    with its "diff --git" header line.
    """
    chunks = []
    for chunk in ("\n" + diff_text).split("\ndiff --git "):
        if chunk.strip():
            chunks.append("diff --git " + chunk.rstrip("\n"))
    return chunks


def compress_diff(diff_text):
    """
    Prepares a diff for the LLM without losing track of the changed files:

    - Per-file diffs larger than MAX_FILE_DIFF_BYTES (e.g. regenerated
      fixtures) are replaced by a placeholder, so a single bulky file
      can't push the rest of the PR past the overall size limit.
    - If the result still exceeds MAX_DIFF_BYTES, it is truncated.

    Returns a tuple (changed_files, compressed_diff, truncated).
    """
    changed_files = []
    parts = []
    for chunk in split_diff_by_file(diff_text):
        header_line = chunk.split("\n", 1)[0]
        changed_files.append(path_from_diff_header(header_line))
        chunk_size = len(chunk.encode())
        if chunk_size > MAX_FILE_DIFF_BYTES:
            line_count = chunk.count("\n")
            parts.append(
                f"{header_line}\n"
                f"[diff omitted: {line_count} lines / {chunk_size} bytes — "
                f"too large for review]"
            )
            log(
                f"Omitting diff of {changed_files[-1]} "
                f"({chunk_size} bytes > {MAX_FILE_DIFF_BYTES})."
            )
        else:
            parts.append(chunk)

    compressed = "\n\n".join(parts)
    truncated = False
    if len(compressed.encode()) > MAX_DIFF_BYTES:
        compressed = compressed.encode()[:MAX_DIFF_BYTES].decode(errors="replace")
        truncated = True
        log(f"Diff truncated to {MAX_DIFF_BYTES} bytes for LLM input.")
    return changed_files, compressed, truncated


def build_user_message(owner, repo, pr, commits, diff):
    current_labels = ', '.join(
        label['name'] for label in pr.get('labels', [])) or '(none)'

    commit_lines = []
    for commit in commits:
        sha = commit.get("sha", "")[:8]
        message = commit.get("commit", {}).get("message", "").rstrip()
        commit_lines.append(f"--- commit {sha} ---\n{message}")

    changed_files, diff_text, truncated = compress_diff(diff)

    parts = [
        '# General information',
        f'Repository:  {owner}/{repo}',
        f'PR #{pr["number"]}:  {pr["title"]}',
        f'State:  {pr["state"]}',
        f'Author:  {pr["user"]["login"]}',
        f'Created:  {pr["created_at"]}',
        f'Labels:  {current_labels}',
        '# PR body:\n\n' + (pr.get('body') or '(empty)'),
        '# Commits:\n\n' + '\n\n'.join(commit_lines),
        '# List of files changed:\n\n' + '\n'.join(f'- {path}' for path in changed_files),
        '# Diff:\n\n' + diff_text,
    ]

    if truncated:
        parts.append(f"[diff truncated to {MAX_DIFF_BYTES} bytes]")

    return '\n\n'.join(parts)


def call_llm(model, user_message, system_prompt=SYSTEM_PROMPT):
    headers = {'Content-Type': 'application/json'}
    if LLM_API_KEY:
        headers['Authorization'] = f'Bearer {LLM_API_KEY}'
    url = LLM_BASE_URL.rstrip('/') + '/v1/chat/completions'
    log(f'Calling {model} at {url} ...')
    response = requests.post(url, headers=headers, json={
        'model': model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_message},
        ],
    }, timeout=LLM_TIMEOUT)
    if response.status_code != 200:
        die(f'LLM endpoint returned {response.status_code}: '
            f'{response.text[:300]}')
    data = response.json()
    content = data.get('choices', [{}])[0].get('message', {}).get('content')
    if not content:
        die(f'LLM response contained no content: {str(data)[:300]}')
    return content.strip()


def upsert_comment(owner, repo, number, comments, body):
    existing = next((comment for comment in comments
                     if COMMENT_MARKER in (comment.get('body') or '')), None)
    if existing:
        url = (f'{GITHUB_API}/repos/{owner}/{repo}/issues/comments/'
               f'{existing["id"]}')
        response = requests.patch(url, headers=gh_headers(),
                                  json={'body': body},
                                  timeout=GITHUB_TIMEOUT)
        action = 'updated'
    else:
        url = f'{GITHUB_API}/repos/{owner}/{repo}/issues/{number}/comments'
        response = requests.post(url, headers=gh_headers(),
                                 json={'body': body},
                                 timeout=GITHUB_TIMEOUT)
        action = 'created'
    if response.status_code not in (200, 201):
        die(f'Posting comment failed with {response.status_code}: '
            f'{response.text[:300]}')
    log(f'Review comment {action}: {response.json().get("html_url", url)}')


def main():
    parser = argparse.ArgumentParser(
        description='Review a GitHub Pull Request with a self-hosted LLM')
    parser.add_argument('repo', help='Repository as owner/name')
    parser.add_argument('pr', type=int, help='PR number')
    parser.add_argument('--post', action='store_true',
                        help='Post/update the review as a PR comment '
                             '(default: print to stdout)')
    parser.add_argument('--show-prompt', action='store_true',
                        help='Print the assembled user prompt and exit without '
                             'calling the LLM')
    parser.add_argument('--model', default=LLM_MODEL,
                        help=f'Model name (default: {LLM_MODEL})')
    parser.add_argument('--prompt-path', default=PROMPT_PATH,
                        help='Path of the custom system prompt in the '
                        f'repository (default: {PROMPT_PATH})')
    parser.add_argument('--stdin-prompt', action='store_true',
                        help='Read the system prompt from stdin instead of '
                        'from the repository or using the generic fallback')
    args = parser.parse_args()

    if '/' not in args.repo:
        die('Repository must be given as owner/name')
    owner, repo = args.repo.split('/', 1)

    if args.post and not GITHUB_TOKEN:
        die('--post requires GITHUB_TOKEN')

    log(f'Fetching PR {owner}/{repo}#{args.pr} ...')
    pr = fetch_pr(owner, repo, args.pr)
    diff = fetch_diff(owner, repo, args.pr)
    commits = fetch_commits(owner, repo, args.pr)
    log(f'PR: "{pr["title"]}" — {pr["commits"]} commit(s), '
        f'{pr["changed_files"]} changed file(s), '
        f'{pr["additions"]} addition(s), {pr["deletions"]} deletion(s)')

    if args.stdin_prompt:
        system_prompt = sys.stdin.read()
    elif p := fetch_prompt(owner, repo, args.prompt_path):
        system_prompt = p
        log(f'Using system prompt from repo: {PROMPT_PATH}')
    else:
        system_prompt = SYSTEM_PROMPT

    user_message = build_user_message(owner, repo, pr, commits, diff)

    if args.show_prompt:
        print(user_message)
        return

    review = call_llm(args.model, user_message, system_prompt=system_prompt)

    if args.post:
        comments = fetch_comments(owner, repo, args.pr)
        body = (f'{COMMENT_MARKER}\n### LLM PR Review ({args.model})\n\n'
                f'{review}')
        upsert_comment(owner, repo, args.pr, comments, body)
    else:
        print(review)


if __name__ == '__main__':
    try:
        main()
    except requests.RequestException as error:
        die(f'Network error: {error}')
