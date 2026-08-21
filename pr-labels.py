#!/usr/bin/env python3
"""Suggest labels for a GitHub pull request with a self-hosted LLM.

Fetches the PR (description, commit messages, diff) and the repository's
label set, asks the model which labels to add or remove, and prints the
suggestions (default) or applies the additions (--apply). Removals are
only ever suggested, never applied.

Usage:
  python3 pr-labels.py digitalfabrik/lunes-cms 1234
  python3 pr-labels.py digitalfabrik/lunes-cms 1234 --apply

Environment:
  LLM_BASE_URL  OpenAI-compatible endpoint base (default http://localhost:11434)
  LLM_API_KEY   Bearer token for the LLM endpoint (optional, e.g. for LiteLLM)
  LLM_MODEL     Model name (default gemma4:31b)
  GITHUB_TOKEN  GitHub API token (optional for reading public repos,
                required for --apply)
"""

import argparse
import json
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
MAX_DIFF_BYTES = 50_000

SYSTEM_PROMPT = """\
You label GitHub pull requests. Given a pull request (title, description,
commit messages, diff) and the repository's available labels, decide which
labels belong on it.

Respond with ONLY a JSON object, no other text:
{"add": [{"label": "<name>", "reason": "<one short sentence>"}],
 "remove": [{"label": "<name>", "reason": "<one short sentence>"}]}

Rules:
- Use ONLY label names from the provided list.
- "add": labels not currently on the PR that clearly apply.
- "remove": current labels that clearly do not apply.
- Judge from the actual changes, not just the title.
- When unsure about a label, leave it out. Empty arrays are fine.
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
        batch = gh_get(url, params={'per_page': 100, 'page': page}).json()
        results.extend(batch)
        if len(batch) < 100:
            return results
        page += 1


def fetch_pr(owner, repo, number):
    return gh_get(f'{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}').json()


def fetch_diff(owner, repo, number):
    diff = gh_get(f'{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}',
                  accept='application/vnd.github.diff').text
    encoded = diff.encode()
    if len(encoded) > MAX_DIFF_BYTES:
        diff = (encoded[:MAX_DIFF_BYTES].decode(errors='replace')
                + '\n\n[Diff was truncated]')
    return diff


def fetch_commits(owner, repo, number):
    return gh_get_paginated(
        f'{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}/commits')


def fetch_labels(owner, repo):
    return gh_get_paginated(f'{GITHUB_API}/repos/{owner}/{repo}/labels')


def build_user_message(owner, repo, pr, commits, diff, labels):
    label_lines = []
    for label in labels:
        description = label.get('description') or ''
        label_lines.append(f'- {label["name"]}: {description}'.rstrip(': '))

    current_labels = ', '.join(
        label['name'] for label in pr.get('labels', [])) or '(none)'
    commit_messages = '\n'.join(
        f'- {commit["commit"]["message"].splitlines()[0]}'
        for commit in commits)

    return '\n\n'.join([
        f'Repository: {owner}/{repo}',
        'Available labels:\n' + '\n'.join(label_lines),
        f'Pull request #{pr["number"]}: {pr["title"]}\n'
        f'Author: {pr["user"]["login"]}\n'
        f'Branch: {pr["head"]["ref"]} -> {pr["base"]["ref"]}\n'
        f'Current labels: {current_labels}',
        'Description:\n\n' + (pr.get('body') or '(empty)'),
        'Commit messages:\n' + commit_messages,
        'Diff:\n\n' + diff,
    ])


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
    content = (response.json().get('choices', [{}])[0]
               .get('message', {}).get('content', '')).strip()
    content = re.sub(r'^```(json)?\s*|\s*```$', '', content)
    try:
        suggestions = json.loads(content)
    except ValueError:
        die(f'LLM did not return valid JSON: {content[:300]}')
    return suggestions


def valid_suggestions(items, known_names):
    results = []
    for item in items or []:
        name = item.get('label')
        if name not in known_names:
            log(f'Ignoring unknown label suggested by model: {name!r}')
            continue
        results.append((name, item.get('reason', '')))
    return results


def apply_labels(owner, repo, number, names):
    url = f'{GITHUB_API}/repos/{owner}/{repo}/issues/{number}/labels'
    response = requests.post(url, headers=gh_headers(),
                             json={'labels': names},
                             timeout=GITHUB_TIMEOUT)
    if response.status_code != 200:
        die(f'Applying labels failed with {response.status_code}: '
            f'{response.text[:300]}')
    log(f'Applied label(s): {", ".join(names)}')


def main():
    parser = argparse.ArgumentParser(
        description='Suggest labels for a GitHub pull request with a '
                    'self-hosted LLM')
    parser.add_argument('repo', help='Repository as owner/name')
    parser.add_argument('pr', type=int, help='Pull request number')
    parser.add_argument('--apply', action='store_true',
                        help='Apply suggested label additions to the PR '
                             '(default: print suggestions only)')
    parser.add_argument('--show-prompt', action='store_true',
                        help='Print the assembled prompt and exit without '
                             'calling the LLM')
    parser.add_argument('--model', default=LLM_MODEL,
                        help=f'Model name (default: {LLM_MODEL})')
    args = parser.parse_args()

    if '/' not in args.repo:
        die('Repository must be given as owner/name')
    owner, repo = args.repo.split('/', 1)

    if args.apply and not GITHUB_TOKEN:
        die('--apply requires GITHUB_TOKEN')

    log(f'Fetching PR {owner}/{repo}#{args.pr} ...')
    pr = fetch_pr(owner, repo, args.pr)
    commits = fetch_commits(owner, repo, args.pr)
    diff = fetch_diff(owner, repo, args.pr)
    labels = fetch_labels(owner, repo)
    log(f'PR: "{pr["title"]}" — {len(commits)} commit(s), '
        f'{len(labels)} repo label(s)')

    user_message = build_user_message(owner, repo, pr, commits, diff, labels)

    if args.show_prompt:
        print(user_message)
        return

    suggestions = call_llm(args.model, user_message)
    known_names = {label['name'] for label in labels}
    current_names = {label['name'] for label in pr.get('labels', [])}
    to_add = [(name, reason)
              for name, reason in valid_suggestions(
                  suggestions.get('add'), known_names)
              if name not in current_names]
    to_remove = [(name, reason)
                 for name, reason in valid_suggestions(
                     suggestions.get('remove'), known_names)
                 if name in current_names]

    if not to_add and not to_remove:
        print('Labels look correct, no changes suggested.')
        return

    for name, reason in to_add:
        print(f'+ {name}: {reason}')
    for name, reason in to_remove:
        print(f'- {name}: {reason} (suggestion only, never auto-removed)')

    if args.apply and to_add:
        apply_labels(owner, repo, args.pr, [name for name, _ in to_add])


if __name__ == '__main__':
    try:
        main()
    except requests.RequestException as error:
        die(f'Network error: {error}')
