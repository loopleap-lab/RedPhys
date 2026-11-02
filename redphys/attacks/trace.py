"""Per-sample optimization trace: attack loops call log(step, ...) every EVERY steps; run.py saves take() per sample.
A step that does not increase starts a new phase (e.g. PA-Attack's two stages)."""

EVERY = 10
_rows: list[dict] = []


def log(step: int, **values) -> None:
    phase = _rows[-1].get("phase", 0) + (step <= _rows[-1]["step"]) if _rows else 0
    _rows.append({"step": int(step), **({"phase": phase} if phase else {}), **{k: round(float(v), 6) for k, v in values.items()}})


def take() -> list[dict]:
    rows = _rows[:]
    _rows.clear()
    return rows
