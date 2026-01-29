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

    output_path = Path(os.environ["GITHUB_OUTPUT"])
    output_path.write_text(
        f"fetch_ok={fetch_ok}\n"
        f"flag={flag}\n"
        f"issue_needed={issue_needed}\n"
        f"issue_title={issue_title}\n"
        "issue_body<<EOF\n"
        f"{issue_body}\n"
        "EOF\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
