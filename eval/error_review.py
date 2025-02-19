"""Prepare a blinded manual sample; do not pretend a heuristic is a native reviewer."""

import argparse
import csv
import json
import random
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("predictions", type=Path)
    p.add_argument("--out", type=Path, default=Path("report/error-review.csv"))
    a = p.parse_args()
    rows = [json.loads(x) for x in a.predictions.read_text().splitlines()]
    selected = random.Random(2026).sample(rows, min(50, len(rows)))
    with a.out.open("w") as stream:
        fields = [
            "id",
            "reference",
            "hypothesis",
            "dropped_tones",
            "wrong_tones",
            "code_switching",
            "wrong_words",
            "reviewer",
            "notes",
        ]
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
