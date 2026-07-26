#!/usr/bin/env python3
"""Find likely duplicates of a GitHub issue via embeddings.

Embeds all open issues of the repository (cached in .cache/) and ranks
them by cosine similarity against the given issue.

Usage:
  python3 issue-duplicates.py digitalfabrik/lunes-cms 914
  python3 issue-duplicates.py digitalfabrik/lunes-cms 914 --top 10 --min-score 0.6

With --llm, a chat model ranks the candidates instead — for endpoints
that offer no embedding model.

Environment:
  LLM_BASE_URL  OpenAI-compatible endpoint base (default http://localhost:11434)
  LLM_API_KEY   Bearer token for the LLM endpoint (optional, e.g. Open WebUI)
  EMBED_MODEL   Embedding model name (default nomic-embed-text)
  LLM_MODEL     Chat model for --llm mode (default gemma4:31b)
  GITHUB_TOKEN  GitHub API token (optional for reading public repos)
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import requests

GITHUB_API = 'https://api.github.com'
LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'http://localhost:11434')
LLM_API_KEY = os.environ.get('LLM_API_KEY', '')
EMBED_MODEL = os.environ.get('EMBED_MODEL', 'nomic-embed-text')
LLM_MODEL = os.environ.get('LLM_MODEL', 'gemma4:31b')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')

GITHUB_TIMEOUT = 30
EMBED_TIMEOUT = 120
LLM_TIMEOUT = 600
MAX_EMBED_CHARS = 8_000
MAX_CANDIDATE_CHARS = 300
BATCH_SIZE = 16
CACHE_DIR = Path('.cache')

LLM_SYSTEM_PROMPT = """\
You detect duplicate GitHub issues. Given a target issue and a list of
candidate issues, identify candidates that likely describe the same
problem or request as the target.

Respond with ONLY a JSON array, no other text. Each element:
{"number": <candidate issue number>, "confidence": <float 0.0-1.0>,
 "reason": "<one short sentence>"}
Include only candidates with confidence >= 0.3. Return [] if none match.
Only use issue numbers that appear in the candidate list.
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


def gh_get_paginated(url, params=None):
    results = []
    page = 1
    while True:
        page_params = dict(params or {}, per_page=100, page=page)
        batch = gh_get(url, params=page_params)
        results.extend(batch)
        if len(batch) < 100:
            return results
        page += 1


def fetch_issue(owner, repo, number):
    issue = gh_get(f'{GITHUB_API}/repos/{owner}/{repo}/issues/{number}')
    if 'pull_request' in issue:
        die(f'#{number} is a pull request, not an issue')
    return issue


def fetch_open_issues(owner, repo):
    issues = gh_get_paginated(f'{GITHUB_API}/repos/{owner}/{repo}/issues',
                              params={'state': 'open'})
    return [issue for issue in issues if 'pull_request' not in issue]


def issue_text(issue):
    text = f'{issue["title"]}\n\n{issue.get("body") or ""}'
    return text[:MAX_EMBED_CHARS]


def embed_texts(texts):
    headers = {'Content-Type': 'application/json'}
    if LLM_API_KEY:
        headers['Authorization'] = f'Bearer {LLM_API_KEY}'
    url = LLM_BASE_URL.rstrip('/') + '/v1/embeddings'
    embeddings = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        response = requests.post(url, headers=headers, json={
            'model': EMBED_MODEL,
            'input': batch,
        }, timeout=EMBED_TIMEOUT)
        if response.status_code != 200:
            die(f'Embeddings endpoint returned {response.status_code}: '
                f'{response.text[:300]}')
        data = response.json().get('data', [])
        if len(data) != len(batch):
            die(f'Expected {len(batch)} embeddings, got {len(data)}')
        embeddings.extend(
            item['embedding'] for item in sorted(data, key=lambda i: i['index']))
        log(f'Embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}')
    return embeddings


def one_line(text, limit):
    return re.sub(r'\s+', ' ', text or '').strip()[:limit]


def llm_duplicates(target, candidates):
    candidate_lines = '\n'.join(
        f'#{issue["number"]}: {issue["title"]} — '
        f'{one_line(issue.get("body"), MAX_CANDIDATE_CHARS)}'
        for issue in candidates)
    user_message = (
        f'Target issue #{target["number"]}: {target["title"]}\n\n'
        f'{one_line(target.get("body"), MAX_EMBED_CHARS)}\n\n'
        f'Candidate issues:\n{candidate_lines}')

    headers = {'Content-Type': 'application/json'}
    if LLM_API_KEY:
        headers['Authorization'] = f'Bearer {LLM_API_KEY}'
    url = LLM_BASE_URL.rstrip('/') + '/v1/chat/completions'
    log(f'Asking {LLM_MODEL} at {url} ...')
    response = requests.post(url, headers=headers, json={
        'model': LLM_MODEL,
        'messages': [
            {'role': 'system', 'content': LLM_SYSTEM_PROMPT},
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
        matches = json.loads(content)
    except ValueError:
        die(f'LLM did not return valid JSON: {content[:300]}')
    by_number = {issue['number']: issue for issue in candidates}
    results = []
    for match in matches:
        issue = by_number.get(match.get('number'))
        if issue is None:
            continue
        results.append((float(match.get('confidence', 0)), issue,
                        match.get('reason', '')))
    return sorted(results, key=lambda item: item[0], reverse=True)


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def cache_path(owner, repo):
    return CACHE_DIR / f'embeddings-{owner}-{repo}.json'


def load_cache(owner, repo):
    path = cache_path(owner, repo)
    if not path.exists():
        return {}
    try:
        cache = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if cache.get('model') != EMBED_MODEL:
        log(f'Embedding model changed, discarding cache {path}')
        return {}
    return cache.get('issues', {})


def save_cache(owner, repo, issues):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path(owner, repo).write_text(
        json.dumps({'model': EMBED_MODEL, 'issues': issues}))


def get_embeddings(owner, repo, issues, no_cache):
    cached = {} if no_cache else load_cache(owner, repo)
    stale = [issue for issue in issues
             if cached.get(str(issue['number']), {}).get('updated_at')
             != issue['updated_at']]
    if stale:
        log(f'Embedding {len(stale)} new/changed issue(s), '
            f'{len(issues) - len(stale)} from cache')
        vectors = embed_texts([issue_text(issue) for issue in stale])
        for issue, vector in zip(stale, vectors):
            cached[str(issue['number'])] = {
                'updated_at': issue['updated_at'],
                'embedding': vector,
            }
        save_cache(owner, repo, cached)
    else:
        log(f'All {len(issues)} embeddings from cache')
    return {issue['number']: cached[str(issue['number'])]['embedding']
            for issue in issues}


def main():
    parser = argparse.ArgumentParser(
        description='Find likely duplicates of a GitHub issue via embeddings')
    parser.add_argument('repo', help='Repository as owner/name')
    parser.add_argument('issue', type=int, help='Issue number')
    parser.add_argument('--top', type=int, default=5,
                        help='Number of matches to show (default: 5)')
    parser.add_argument('--min-score', type=float, default=0.5,
                        help='Minimum similarity to report (default: 0.5)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Ignore and rebuild the embedding cache')
    parser.add_argument('--llm', action='store_true',
                        help='Rank candidates with the chat model instead of '
                             'embeddings (for endpoints without an embedding '
                             'model)')
    args = parser.parse_args()

    if '/' not in args.repo:
        die('Repository must be given as owner/name')
    owner, repo = args.repo.split('/', 1)

    log(f'Fetching issue {owner}/{repo}#{args.issue} ...')
    target = fetch_issue(owner, repo, args.issue)
    log('Fetching open issues ...')
    candidates = [issue for issue in fetch_open_issues(owner, repo)
                  if issue['number'] != target['number']]
    log(f'{len(candidates)} open issue(s) to compare against')

    if args.llm:
        results = llm_duplicates(target, candidates)
        matches = [(score, issue, reason)
                   for score, issue, reason in results[:args.top]
                   if score >= args.min_score]
        print(f'Similar open issues for #{target["number"]} '
              f'"{target["title"]}" (model: {LLM_MODEL}):\n')
        if not matches:
            print(f'No matches with confidence >= {args.min_score}')
            return
        for score, issue, reason in matches:
            print(f'{score:.2f}  #{issue["number"]:<5}  {issue["title"]}\n'
                  f'        {issue["html_url"]}\n'
                  f'        {reason}')
        return

    embeddings = get_embeddings(owner, repo, [target] + candidates,
                                args.no_cache)

    target_vector = embeddings[target['number']]
    scored = sorted(
        ((cosine(target_vector, embeddings[issue['number']]), issue)
         for issue in candidates),
        key=lambda pair: pair[0], reverse=True)

    matches = [(score, issue) for score, issue in scored[:args.top]
               if score >= args.min_score]
    print(f'Similar open issues for #{target["number"]} '
          f'"{target["title"]}" (model: {EMBED_MODEL}):\n')
    if not matches:
        print(f'No matches with similarity >= {args.min_score}')
        if scored:
            score, issue = scored[0]
            print(f'Closest was {score:.3f}  #{issue["number"]}  '
                  f'{issue["title"]}')
        return
    for score, issue in matches:
        print(f'{score:.3f}  #{issue["number"]:<5}  {issue["title"]}\n'
              f'        {issue["html_url"]}')


if __name__ == '__main__':
    try:
        main()
    except requests.RequestException as error:
        die(f'Network error: {error}')
