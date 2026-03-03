#!/usr/bin/env python3
"""Create or update a GitHub Issue based on workflow status."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def decode_response(headers: object, raw: bytes) -> str:
    """Decode HTTP payload for diagnostics; prefer UTF-8 then latin-1."""
    charset_getter = getattr(headers, "get_content_charset", None)
    charset = charset_getter() if callable(charset_getter) else None
    for encoding in (charset, "utf-8", "latin-1"):
        if not encoding:
            continue
        try:
            return raw.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def request(url: str, method: str = "GET", payload: dict | None = None, headers: dict | None = None) -> dict:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
            if "json" not in content_type.lower():
                preview = decode_response(response.headers, raw)[:300]
                raise RuntimeError(f"Expected JSON but received {content_type!r}: {preview}")
            return json.loads(raw)
    except urllib.error.HTTPError as error:
        raw = error.read()
        preview = decode_response(error.headers, raw)[:300]
        raise RuntimeError(f"GitHub API request failed ({error.code} {error.reason}): {preview}") from error


def main() -> int:
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    title = os.environ["ISSUE_TITLE"]
    body = os.environ["ISSUE_BODY"]

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "gold-risk-monitor",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    issues_url = f"https://api.github.com/repos/{repo}/issues?state=open&per_page=100"
    issues = request(issues_url, headers=headers)
    issue = next((item for item in issues if item.get("title") == title and "pull_request" not in item), None)

    payload = {"title": title, "body": body}
    if issue:
        updated = request(issue["url"], method="PATCH", payload=payload, headers=headers)
        issue_url = updated.get("html_url", "")
    else:
        created = request(f"https://api.github.com/repos/{repo}/issues", method="POST", payload=payload, headers=headers)
        issue_url = created.get("html_url", "")
    if not issue_url:
        raise RuntimeError("Create or update issue step failed (missing issue URL).")

    env_file = os.environ.get("GITHUB_ENV", "").strip()
    if not env_file:
        raise RuntimeError("GITHUB_ENV is not set")

    with open(env_file, "a", encoding="utf-8") as handle:
        handle.write(f"ISSUE_URL={issue_url}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
