#!/usr/bin/env python3
"""Create or update a GitHub Issue based on workflow status."""
from __future__ import annotations

import json
import os
import urllib.request


def request(url: str, method: str = "GET", payload: dict | None = None, headers: dict | None = None) -> dict:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    title = os.environ["ISSUE_TITLE"]
    body = os.environ["ISSUE_BODY"]

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "gold-risk-monitor",
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

    print(f"::set-output name=issue_url::{issue_url}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
