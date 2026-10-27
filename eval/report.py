import argparse
import csv
import json
from pathlib import Path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("report/results"))
    args = parser.parse_args()
    rows = [json.loads(p.read_text()) for p in args.summaries]
    if any(r.get("status") != "measured" for r in rows):
        raise ValueError("Reports accept only measured runs")
    args.out.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "label",
        "backend",
        "compute_type",
        "n",
        "partial",
        "wer",
        "tone_error_rate",
        "reference_normalized_tone_edit_rate",
        "conditional_tone_error_rate",
        "matched_word_coverage",
        "matched_tone_coverage",
        "reference_tones",
        "matched_reference_tones",
        "tone_edits",
        "source_revision",
        "evaluation_set_sha256",
        "real_time_factor",
        "cost_per_audio_hour_usd",
    ]
    with (args.out / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = 0
    for row in rows:
        cost, ter = row.get("cost_per_audio_hour_usd"), row.get("tone_error_rate")
        if cost is None or ter is None:
            continue
        label = row.get("label") or f"{row['model']} ({row['compute_type']})"
        size, _, precision = label.partition("-")
        colors = {
            "ct2-float32": "#64748b",
            "ct2-int8": "#16836b",
            "cpp-f16": "#4c78a8",
            "cpp-q5_0": "#b77924",
            "cpp-q4_0": "#a34561",
        }
        ax.scatter(
            cost,
            ter * 100,
            label=label,
            marker={"tiny": "o", "base": "s", "small": "^"}.get(size, "o"),
            color=colors.get(precision),
            s=48,
        )
        plotted += 1
    if not plotted:
        ax.text(
            0.5,
            0.5,
            "Cloud price evidence pending; no cost points available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set(
        xlabel="USD per audio hour",
        ylabel="Conditional explicit-tone error rate (%)",
        title="Yorùbá ASR: measured cost and tone accuracy",
    )
    if plotted:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7, frameon=False)
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(args.out / "cost-tone.png", dpi=180)


if __name__ == "__main__":
    main()
