"""
utils/data_dir.py - Shared path helpers for the data/{N}_elastics/ directory layout.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Step normalisation constants - shared between model training (server) and
# ONNX inference (Pi). The aim head outputs values in [-1, 1] which are
# scaled by these at inference time.
# ---------------------------------------------------------------------------

NORMALISE_X: float = 10_000.0
NORMALISE_Y: float = 20_000.0


def elastics_data_dir(elastics: int) -> Path:
    """Return the data directory for the given elastic count."""
    return Path("data") / f"{elastics}_elastics"


def checkpoint_path(data_dir: Path, policy_name: str) -> Path:
    """Return the checkpoint file path for a given policy (e.g. gru.pt, heuristic.json)."""
    ext = ".pt" if policy_name in ("gru", "outer_gru") else ".json"
    return data_dir / "checkpoints" / f"{policy_name}{ext}"


# ---------------------------------------------------------------------------
# Cup grid position parsing
# ---------------------------------------------------------------------------


def parse_cup_arg(cup_str: str | None) -> tuple[int | None, int | None]:
    """Parse a '--cup X,Y' argument into (cup_x, cup_y). Returns (None, None) if cup_str is None."""
    if cup_str is None:
        return None, None
    parts = cup_str.split(",")
    if len(parts) != 2:
        raise ValueError(f"Expected X,Y format, got: {cup_str!r}")
    cx, cy = int(parts[0].strip()), int(parts[1].strip())
    if cx < 1 or cy < 1:
        raise ValueError(f"Cup coordinates must be positive, got: ({cx},{cy})")
    return cx, cy


def parse_cup_list(cups: tuple[str, ...]) -> list[tuple[int, int]]:
    """
    Parse multiple '--cup X,Y' arguments for outer sessions.

    Returns a list of (cup_x, cup_y) tuples. All must be valid.
    Raises ValueError on any invalid entry.
    """
    result: list[tuple[int, int]] = []
    for c in cups:
        cx, cy = parse_cup_arg(c)
        if cx is None or cy is None:
            raise ValueError(f"Invalid cup position: {c!r}")
        result.append((cx, cy))
    return result


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file. Returns [] if file doesn't exist."""
    import json

    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
