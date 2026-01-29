#!/usr/bin/env python3
"""Attach latest issue URL to data.json."""
from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    path = Path("data.json")
    if not path.exists():
        return 0

    issue_url = os.environ.get("ISSUE_URL", "")
    if not issue_url:
        return 0

    data = json.loads(path.read_text())
    data["latest_issue_url"] = issue_url
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
