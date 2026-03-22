import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def append_metrics_log(root: Path, name: str, payload: dict[str, Any]) -> None:
    write_json(root / "metrics" / f"{name}.json", payload)
