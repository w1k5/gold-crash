#!/usr/bin/env python3
"""Read updater status JSON and emit GitHub Actions outputs."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    status_path = Path("/tmp/monitor_status.json")
    status = json.loads(status_path.read_text())
    fetch_ok = "true" if status.get("fetch_ok") else "false"
    flag = status.get("flag") or ""
    issue_title = status.get("issue_title") or ""
    issue_body = status.get("issue_body") or ""
    issue_needed = "true" if issue_title and issue_body else "false"

    print(f"::set-output name=fetch_ok::{fetch_ok}")
    print(f"::set-output name=flag::{flag}")
    print(f"::set-output name=issue_needed::{issue_needed}")
    print(f"::set-output name=issue_title::{issue_title}")
    print(f"::set-output name=issue_body::{issue_body}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
