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
        "backend",
        "compute_type",
        "n",
        "wer",
        "tone_error_rate",
        "conditional_tone_error_rate",
        "matched_tone_coverage",
        "real_time_factor",
        "cost_per_audio_hour_usd",
    ]
    with (args.out / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for row in rows:
        cost, ter = row.get("cost_per_audio_hour_usd"), row.get("tone_error_rate")
        if cost is None or ter is None:
            continue
        ax.scatter(cost, ter * 100)
        ax.annotate(f"{row['model']} ({row['compute_type']})", (cost, ter * 100), fontsize=8)
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
        ylabel="Explicit tone error rate (%)",
        title="Yorùbá ASR: measured cost and tone accuracy",
    )
    fig.tight_layout()
    fig.savefig(args.out / "cost-tone.png", dpi=180)


if __name__ == "__main__":
    main()
