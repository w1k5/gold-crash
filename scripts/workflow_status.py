#!/usr/bin/env python3
"""Read updater status JSON and emit GitHub Actions outputs."""
from __future__ import annotations

import json
import os
from pathlib import Path


def write_env_var(handle, key: str, value: str) -> None:
    if "\n" in value:
        delimiter = "__GOLD_RISK_MONITOR__"
        handle.write(f"{key}<<{delimiter}\n{value}\n{delimiter}\n")
        return
    handle.write(f"{key}={value}\n")


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
        write_env_var(handle, "FETCH_OK", fetch_ok)
        write_env_var(handle, "FLAG", flag)
        write_env_var(handle, "ISSUE_NEEDED", issue_needed)
        write_env_var(handle, "ISSUE_TITLE", issue_title)
        write_env_var(handle, "ISSUE_BODY", issue_body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
