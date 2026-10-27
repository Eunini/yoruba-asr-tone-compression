"""Fail-closed resumption of a fixed ASR experiment, without rewriting answers."""

import json
import math
from pathlib import Path


def prepare(out: Path, identity: dict, rows: list[dict], resume: bool) -> list[dict]:
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Evaluation IDs must be unique")
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / "run-identity.json"
    predictions = out / "predictions.jsonl"
    if receipt.exists():
        if not resume:
            raise FileExistsError("Run already exists; use --resume with identical inputs")
        if json.loads(receipt.read_text()) != identity:
            raise ValueError("Source, model, data or decoding identity changed; use a new run")
    else:
        if predictions.exists() or (out / "summary.json").exists():
            raise ValueError("Cannot resume legacy results without a recorded run identity")
        # Exclusive creation avoids silently replacing a competing runner's identity.
        with receipt.open("x") as stream:
            json.dump(identity, stream, indent=2)
            stream.write("\n")
    if not predictions.exists():
        return []
    raw = predictions.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Incomplete prediction tail; preserve and investigate before resuming")
    saved = [json.loads(line) for line in raw.splitlines()]
    if len(saved) > len(rows):
        raise ValueError("Saved predictions exceed the evaluation set")
    for prediction, row in zip(saved, rows, strict=False):
        if any(prediction.get(k) != row[k] for k in ("id", "reference", "duration_seconds")):
            raise ValueError("Saved prediction does not match the ordered evaluation prefix")
        elapsed = prediction.get("processing_seconds")
        if (
            not isinstance(prediction.get("hypothesis"), str)
            or not isinstance(elapsed, (float, int))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError("Saved prediction is invalid")
    return saved
