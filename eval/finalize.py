"""Publish only complete CPU matrices; keep scientific and human-review gates explicit."""

import argparse
import json
import hashlib
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from eval.artifacts import fingerprint
from eval.compression import TEST_EVALUATION_SHA256
from eval.metrics import score
from eval.paired import compare


def complete_summaries(matrix):
    progress = json.loads((matrix / "progress.json").read_text())
    labels = {
        size + "-" + variant
        for size in ("tiny", "base", "small")
        for variant in ("ct2-float32", "ct2-int8", "cpp-f16", "cpp-q5_0", "cpp-q4_0")
    }
    if (
        progress.get("status") != "complete"
        or len(progress.get("completed_conditions", [])) != 15
        or set(progress["completed_conditions"]) != labels
    ):
        raise ValueError("Only a complete 15-condition matrix may be finalized")
    summaries = [matrix / name / "summary.json" for name in progress["completed_conditions"]]
    records = [json.loads(p.read_text()) for p in summaries]
    if {r.get("label") for r in records} != labels or any(
        r.get("status") != "measured"
        or r.get("n") != 177
        or r.get("partial")
        or r.get("cost_per_audio_hour_usd") is None
        for r in records
    ):
        raise ValueError("Every condition needs the full test and measured CPU price evidence")
    if len({r["evaluation_set_sha256"] for r in records}) != 1:
        raise ValueError("Matrix conditions use different evaluation sets")
    if records[0]["evaluation_set_sha256"] != TEST_EVALUATION_SHA256:
        raise ValueError("Use the original audited test corpus")
    predictions = {}
    for record, summary_path in zip(records, summaries, strict=True):
        rows = [
            json.loads(line)
            for line in summary_path.with_name("predictions.jsonl").read_text().splitlines()
        ]
        if len(rows) != 177 or len({r["id"] for r in rows}) != 177:
            raise ValueError("Incomplete or duplicate prediction records")
        evaluation = hashlib.sha256(
            json.dumps(
                [
                    {key: row[key] for key in ("id", "reference", "duration_seconds")}
                    for row in rows
                ],
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if evaluation != record["evaluation_set_sha256"]:
            raise ValueError("Prediction references differ from the recorded evaluation")
        measured = score([r["reference"] for r in rows], [r["hypothesis"] for r in rows])
        if any(record.get(key) != value for key, value in measured.items()):
            raise ValueError("Reported scores do not reproduce from predictions")
        duration = sum(r["duration_seconds"] for r in rows)
        elapsed = sum(r["processing_seconds"] for r in rows)
        if not math.isclose(
            record["cost_per_audio_hour_usd"],
            elapsed / duration * record["server_hourly_usd"],
            rel_tol=1e-12,
        ):
            raise ValueError("Reported serving cost does not reproduce")
        predictions[record["label"]] = rows
    paired = list(matrix.glob("*-paired.json"))
    if len(paired) != 9 or any(json.loads(p.read_text()).get("n") != 177 for p in paired):
        raise ValueError("All nine full-test within-backend comparisons are required")
    for size in ("tiny", "base", "small"):
        for control, quantized in (
            ("ct2-float32", "ct2-int8"),
            ("cpp-f16", "cpp-q5_0"),
            ("cpp-f16", "cpp-q4_0"),
        ):
            result = compare(predictions[size + "-" + control], predictions[size + "-" + quantized])
            stored = json.loads((matrix / (size + "-" + quantized + "-paired.json")).read_text())
            if any(stored.get(key) != value for key, value in result.items()):
                raise ValueError("Paired tone statistics do not reproduce")
    return summaries, records


def publish_owned_paths(repo, paths, push=True):
    paths = [str(Path(path).resolve().relative_to(repo.resolve())) for path in paths]
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repo, text=True
    ).strip()
    if branch != "main":
        raise ValueError("Results saved; automatic publication only uses the main branch")
    if push:
        account = json.loads(subprocess.check_output(["gh", "api", "user"], text=True))["login"]
        if account != "Eunini":
            raise ValueError("Results saved; unexpected GitHub publishing identity")
    subprocess.run(["git", "add", "--", *paths], cwd=repo, check=True)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths], cwd=repo)
    if changed.returncode not in (0, 1):
        raise RuntimeError("Could not check publication changes")
    if changed.returncode == 1:
        # --only excludes unrelated staged user changes from this commit.
        subprocess.run(
            [
                "git",
                "commit",
                "--only",
                "-m",
                "Record complete CPU compression measurements and teardown evidence",
                "--",
                *paths,
            ],
            cwd=repo,
            check=True,
        )
    if push:
        subprocess.run(["git", "push", "origin", "HEAD:main"], cwd=repo, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--publish", action="store_true")
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    matrix = a.matrix.resolve(strict=True)
    if not matrix.is_relative_to(repo / "report/runs"):
        raise ValueError("Use this repository's own collected matrix")
    summaries, records = complete_summaries(matrix)
    controller = json.loads((matrix / "controller.json").read_text())
    if controller.get("teardown_verified") is not True:
        raise ValueError("Verify cloud cleanup before publishing a completed run")
    target = repo / "report/results" / matrix.name
    subprocess.run(
        [sys.executable, "-m", "eval.report", *map(str, summaries), "--out", str(target)],
        cwd=repo,
        check=True,
    )
    digest = fingerprint(target / "results.csv")
    (target / "completion.json").write_text(
        json.dumps(
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "conditions": len(records),
                "utterances_per_condition": 177,
                "paired_comparisons": 9,
                "teardown_verified": True,
                "summary_files": [str(p.relative_to(repo)) for p in summaries],
                "table_sha256": digest.get("results.csv"),
                "scientific_release_status": "CPU matrix complete; manual/native 50-output error review and final interpretation still required",
            },
            indent=2,
        )
        + "\n"
    )
    (target / "README.md").write_text(
        "# Completed CPU compression matrix\n\nAll 15 converted model conditions completed the same 177 recordings. "
        "The table reports WER, conditional TER, coverage, real-time factor and measured list-price cost. "
        "Nine paired analyses in the run directory hold lexical support fixed.\n\n"
        "![Measured cost and conditional tone errors](cost-tone.png)\n\n"
        "The observed Flex-server timing is one operating point. Backend decoding differs; "
        "compare quantization with its own float control. Read TER together with coverage. "
        "This does not establish a general tone-first failure law. The complete study still needs "
        "the manual/native 50-output error review and final interpretation.\n"
    )
    paths = [matrix, target]
    readme = repo / "README.md"
    if not subprocess.check_output(
        ["git", "diff", "HEAD", "--", "README.md"], cwd=repo, text=True
    ).strip():
        text = readme.read_text()
        banner = (
            f"\n**CPU matrix completed:** all 15 converted conditions and nine paired analyses are "
            f"available in [the measured results](report/results/{matrix.name}/README.md). "
            "Cloud teardown was verified. Manual error review and final interpretation remain pending.\n"
        )
        if banner not in text:
            text = text.replace(
                "The compression/cost comparison\nremains pending;",
                "The converted-model CPU matrix is complete;",
                1,
            )
            text = text.replace(
                "the fine-tuned models do not outperform this MMS baseline.",
                "the Hugging Face fine-tuned runs did not outperform the MMS baseline.",
                1,
            )
            title, rest = text.split("\n", 1)
            readme.write_text(title + "\n" + banner + rest)
        paths.append(readme)
    if a.publish:
        publish_owned_paths(repo, paths)
    print("Complete CPU matrix, chart and table saved; cloud teardown verified", flush=True)


if __name__ == "__main__":
    main()
