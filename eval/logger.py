import logging
from pathlib import Path
from datetime import datetime


import logging
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str,
    log_dir: Path,
    level=logging.INFO,
) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{ts}.log"

    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
        )

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)

        root.addHandler(ch)
        root.addHandler(fh)

    return logging.getLogger(name)

