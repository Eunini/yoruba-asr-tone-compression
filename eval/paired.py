"""Compare tone errors on reference word positions recovered by both outputs."""

import argparse
import json
import random
from pathlib import Path

import jiwer

from eval.metrics import edit_distance, normalize, score, strip_tones, tones


def matched_positions(reference, hypothesis):
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    alignment = jiwer.process_words(
        strip_tones(normalize(reference)), strip_tones(normalize(hypothesis))
    )
    positions = {}
    for chunk in alignment.alignments[0]:
        if chunk.type == "equal":
            for offset in range(chunk.ref_end_idx - chunk.ref_start_idx):
                r, h = chunk.ref_start_idx + offset, chunk.hyp_start_idx + offset
                positions[r] = (ref[r], hyp[h])
    return positions


def compare(control, quantized, replicates=5000):
    if replicates < 1 or not control:
        raise ValueError("Provide nonempty complete predictions and positive bootstrap count")
    left = {r["id"]: r for r in control}
    right = {r["id"]: r for r in quantized}
    if len(left) != len(control) or len(right) != len(quantized) or left.keys() != right.keys():
        raise ValueError("Runs must contain the same unique utterance IDs")
    counts = []
    for item_id, row in left.items():
        other = right[item_id]
        if (row["reference"], row["duration_seconds"]) != (
            other["reference"],
            other["duration_seconds"],
        ):
            raise ValueError("Reference text or duration differs between runs")
        a = matched_positions(row["reference"], row["hypothesis"])
        b = matched_positions(row["reference"], other["hypothesis"])
        common = sorted(a.keys() & b.keys())
        counts.append(
            {
                "id": item_id,
                "reference_words": len(normalize(row["reference"]).split()),
                "reference_tones": len(tones(row["reference"])),
                "common_words": len(common),
                "common_reference_tones": sum(len(tones(a[i][0])) for i in common),
                "control_edits": sum(edit_distance(tones(a[i][0]), tones(a[i][1])) for i in common),
                "quantized_edits": sum(
                    edit_distance(tones(b[i][0]), tones(b[i][1])) for i in common
                ),
            }
        )
    totals = {key: sum(r[key] for r in counts) for key in counts[0] if key != "id"}
    denominator = totals["common_reference_tones"]
    delta = (
        (totals["quantized_edits"] - totals["control_edits"]) / denominator if denominator else None
    )
    rng = random.Random(2026)
    samples = []
    for _ in range(replicates):
        selected = rng.choices(counts, k=len(counts))
        n = sum(r["common_reference_tones"] for r in selected)
        if n:
            samples.append(sum(r["quantized_edits"] - r["control_edits"] for r in selected) / n)
    samples.sort()
    interval = [samples[int((len(samples) - 1) * p)] for p in (0.025, 0.975)] if samples else None
    references = [r["reference"] for r in control]
    return {
        "status": "measured",
        "n": len(counts),
        "policy": "intersection of correctly aligned reference-word positions; explicit acute/grave sequences",
        **totals,
        "common_word_coverage": totals["common_words"] / totals["reference_words"]
        if totals["reference_words"]
        else None,
        "common_tone_coverage": denominator / totals["reference_tones"]
        if totals["reference_tones"]
        else None,
        "control_common_ter": totals["control_edits"] / denominator if denominator else None,
        "quantized_common_ter": totals["quantized_edits"] / denominator if denominator else None,
        "common_ter_delta": delta,
        "paired_utterance_bootstrap_95ci": interval,
        "bootstrap_seed": 2026,
        "bootstrap_replicates": replicates,
        "bootstrap_valid_replicates": len(samples),
        "control_marginal": score(references, [r["hypothesis"] for r in control]),
        "quantized_marginal": score(references, [right[r["id"]]["hypothesis"] for r in control]),
        "per_utterance": counts,
        "interpretation": "Within-backend association on fixed lexical support; does not establish that tone errors rise before word errors",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--quantized", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    def read(path):
        return [json.loads(line) for line in path.read_text().splitlines()]

    result = compare(read(a.control), read(a.quantized))
    result.update(control_file=str(a.control), quantized_file=str(a.quantized))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
