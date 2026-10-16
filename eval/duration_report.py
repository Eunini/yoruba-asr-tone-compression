"""Post-hoc short/long test diagnostics; never used for checkpoint selection."""

import argparse
import json
from pathlib import Path

from eval.metrics import score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("report/duration-diagnostic.json"))
    a = p.parse_args()
    output = {
        "scope": "Post-hoc duration diagnostic; not checkpoint selection or a replacement for the full test",
        "bins": {},
        "training": {},
    }
    for directory in a.runs:
        rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
        summary = json.loads((directory / "summary.json").read_text())
        result = score([r["reference"] for r in rows], [r["hypothesis"] for r in rows])
        for metric in ["wer", "tone_error_rate", "matched_tone_coverage"]:
            if abs(result[metric] - summary[metric]) > 1e-10:
                raise ValueError("Prediction/summary discrepancy: " + directory.name)
        groups = {}
        for label, subset in [
            ("up_to_30s", [r for r in rows if r["duration_seconds"] <= 30]),
            ("over_30s", [r for r in rows if r["duration_seconds"] > 30]),
        ]:
            groups[label] = {
                "n": len(subset),
                "audio_seconds": sum(r["duration_seconds"] for r in subset),
                **score([r["reference"] for r in subset], [r["hypothesis"] for r in subset]),
            }
        output["bins"][directory.name] = groups
        selection = directory / "selection.json"
        if selection.exists():
            value = json.loads(selection.read_text())
            history = value["training_history"]
            output["training"][directory.name] = {
                "checkpoint": value["checkpoint"],
                "validation_wer": value["validation_wer"],
                "length_exclusions": value["length_exclusions"],
                "logged_training_steps": len(history),
                "first_training_entry": history[0],
                "last_training_entry": history[-1],
            }
    a.out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
