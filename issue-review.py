#!/usr/bin/env python3
"""Review a GitHub issue with a self-hosted LLM.

Fetches the issue, its comments and the repository's label set, asks the
model for a triage review, and prints it (default) or posts it as an
idempotent marker comment (--post).

Usage:
  python3 issue-review.py digitalfabrik/lunes-cms 914
  python3 issue-review.py digitalfabrik/lunes-cms 914 --post

Environment:
  LLM_BASE_URL  OpenAI-compatible endpoint base (default http://localhost:11434)
  LLM_API_KEY   Bearer token for the LLM endpoint (optional, e.g. for LiteLLM)
  LLM_MODEL     Model name (default gemma4:31b)
  GITHUB_TOKEN  GitHub API token (optional for reading public repos,
                required for --post)
"""

import argparse
import os
import sys

import requests

GITHUB_API = 'https://api.github.com'
LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'http://localhost:11434')
LLM_MODEL = os.environ.get('LLM_MODEL', 'gemma4:31b')
LLM_API_KEY = os.environ.get('LLM_API_KEY', '')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')

GITHUB_TIMEOUT = 30
LLM_TIMEOUT = 600
MAX_COMMENTS_BYTES = 50_000
COMMENT_MARKER = '<!-- llm-issue-review -->'

SYSTEM_PROMPT = """\
You are an experienced open-source maintainer triaging a GitHub issue.
Write a concise triage review with exactly these Markdown sections:

### Classification
Bug report, feature request, task, or question — plus one sentence on what
the issue asks for.

### Completeness
Is the issue actionable as written? List concrete information that is
missing (reproduction steps, versions, acceptance criteria, ...). If it
follows an issue template, note empty or unhelpful sections.

### Labels
The repository's available labels are listed in the user message. Suggest
labels to add or remove, choosing ONLY from that list. Briefly justify
each suggestion. If the current labels are already correct, say so.

### Open questions
Design or scoping questions the issue leaves unanswered that should be
resolved before implementation starts.

### Suggested next step
One sentence, e.g. "ready to implement", "needs maintainer decision on X",
"ask the reporter for Y".

Rules:
- Base everything strictly on the issue text; do not invent details.
- Be specific and terse. No praise, no filler, no restating the whole issue.
- Do not propose implementation code.
"""


def die(message):
    print(f'Error: {message}', file=sys.stderr)
    sys.exit(1)


def log(message):
    print(message, file=sys.stderr)


def gh_headers():
    headers = {'Accept': 'application/vnd.github+json'}
    if GITHUB_TOKEN:
        headers['Authorization'] = f'Bearer {GITHUB_TOKEN}'
    return headers


def gh_get(url, params=None):
    response = requests.get(url, headers=gh_headers(), params=params,
                            timeout=GITHUB_TIMEOUT)
    if response.status_code != 200:
        die(f'GitHub API {url} returned {response.status_code}: '
            f'{response.text[:300]}')
    return response.json()


def gh_get_paginated(url):
    results = []
    page = 1
    while True:
        batch = gh_get(url, params={'per_page': 100, 'page': page})
        results.extend(batch)
        if len(batch) < 100:
            return results
        page += 1


def fetch_issue(owner, repo, number):
    issue = gh_get(f'{GITHUB_API}/repos/{owner}/{repo}/issues/{number}')
    if 'pull_request' in issue:
        die(f'#{number} is a pull request, not an issue')
    return issue


def fetch_comments(owner, repo, number):
    return gh_get_paginated(
        f'{GITHUB_API}/repos/{owner}/{repo}/issues/{number}/comments')


def fetch_labels(owner, repo):
    return gh_get_paginated(f'{GITHUB_API}/repos/{owner}/{repo}/labels')


def build_user_message(owner, repo, issue, comments, labels):
    label_lines = []
    for label in labels:
        description = label.get('description') or ''
        label_lines.append(f'- {label["name"]}: {description}'.rstrip(': '))

    current_labels = ', '.join(
        label['name'] for label in issue.get('labels', [])) or '(none)'

    parts = [
        f'Repository: {owner}/{repo}',
        'Available labels:\n' + '\n'.join(label_lines),
        f'Issue #{issue["number"]}: {issue["title"]}\n'
        f'State: {issue["state"]}\n'
        f'Author: {issue["user"]["login"]}\n'
        f'Created: {issue["created_at"]}\n'
        f'Current labels: {current_labels}',
        'Issue body:\n\n' + (issue.get('body') or '(empty)'),
    ]

    if comments:
        comment_blocks = []
        for comment in comments:
            if COMMENT_MARKER in (comment.get('body') or ''):
                continue
            comment_blocks.append(
                f'--- comment by {comment["user"]["login"]} '
                f'({comment["created_at"]}) ---\n{comment["body"]}')
        comments_text = '\n\n'.join(comment_blocks)
        encoded = comments_text.encode()
        if len(encoded) > MAX_COMMENTS_BYTES:
            comments_text = (encoded[:MAX_COMMENTS_BYTES]
                             .decode(errors='replace')
                             + '\n\n[Comments were truncated]')
        if comments_text:
            parts.append('Comments:\n\n' + comments_text)

    return '\n\n'.join(parts)


def call_llm(model, user_message):
    headers = {'Content-Type': 'application/json'}
    if LLM_API_KEY:
        headers['Authorization'] = f'Bearer {LLM_API_KEY}'
    url = LLM_BASE_URL.rstrip('/') + '/v1/chat/completions'
    log(f'Calling {model} at {url} ...')
    response = requests.post(url, headers=headers, json={
        'model': model,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
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
        description='Review a GitHub issue with a self-hosted LLM')
    parser.add_argument('repo', help='Repository as owner/name')
    parser.add_argument('issue', type=int, help='Issue number')
    parser.add_argument('--post', action='store_true',
                        help='Post/update the review as an issue comment '
                             '(default: print to stdout)')
    parser.add_argument('--show-prompt', action='store_true',
                        help='Print the assembled prompt and exit without '
                             'calling the LLM')
    parser.add_argument('--model', default=LLM_MODEL,
                        help=f'Model name (default: {LLM_MODEL})')
    args = parser.parse_args()

    if '/' not in args.repo:
        die('Repository must be given as owner/name')
    owner, repo = args.repo.split('/', 1)

    if args.post and not GITHUB_TOKEN:
        die('--post requires GITHUB_TOKEN')

    log(f'Fetching issue {owner}/{repo}#{args.issue} ...')
    issue = fetch_issue(owner, repo, args.issue)
    comments = fetch_comments(owner, repo, args.issue)
    labels = fetch_labels(owner, repo)
    log(f'Issue: "{issue["title"]}" — {len(comments)} comment(s), '
        f'{len(labels)} repo label(s)')

    user_message = build_user_message(owner, repo, issue, comments, labels)

    if args.show_prompt:
        print(user_message)
        return

    review = call_llm(args.model, user_message)

    if args.post:
        body = (f'{COMMENT_MARKER}\n### LLM Issue Review ({args.model})\n\n'
                f'{review}')
        upsert_comment(owner, repo, args.issue, comments, body)
    else:
        print(review)


if __name__ == '__main__':
    try:
        main()
    except requests.RequestException as error:
        die(f'Network error: {error}')
