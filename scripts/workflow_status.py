#!/usr/bin/env python3
"""Read updater status JSON and emit GitHub Actions outputs."""
from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    status_path = Path("/tmp/monitor_status.json")
    status = json.loads(status_path.read_text())
    fetch_ok = "true" if status.get("fetch_ok") else "false"
    flag = status.get("flag") or ""
    issue_title = status.get("issue_title") or ""
    issue_body = status.get("issue_body") or ""
    issue_needed = "true" if issue_title and issue_body else "false"

    env_file = os.environ.get("GITHUB_ENV", "").strip()
    if not env_file:
        raise RuntimeError("GITHUB_ENV is not set")

    env_path = Path(env_file)
    with env_path.open("a", encoding="utf-8") as handle:
        handle.write(f"FETCH_OK={fetch_ok}\n")
        handle.write(f"FLAG={flag}\n")
        handle.write(f"ISSUE_NEEDED={issue_needed}\n")
        handle.write(f"ISSUE_TITLE={issue_title}\n")
        handle.write(f"ISSUE_BODY={issue_body}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
