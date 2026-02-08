from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "-m", "QA", "pipeline", *sys.argv[1:]]
    return subprocess.call(cmd, cwd=root.parent)


if __name__ == "__main__":
    raise SystemExit(main())

