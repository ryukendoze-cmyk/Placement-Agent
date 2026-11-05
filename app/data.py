"""Light helpers over data/ files so stubs/tools can detect Phase 3 assets
without importing heavy modules."""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def placement_sops_available() -> bool:
    return (DATA_DIR / "placement_sops.json").exists()


def read_json(name: str):
    path = DATA_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
