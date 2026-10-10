import json
import os
from pathlib import Path
import tempfile


def write_private_json(path: Path, data) -> None:
    """Replace a JSON file atomically, with private permissions and unique staging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)
