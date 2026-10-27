import json
import hashlib
import subprocess

import pytest

from eval.finalize import complete_summaries, publish_owned_paths
from eval.metrics import score
from eval.paired import compare


def test_partial_matrix_cannot_be_presented_as_complete(tmp_path):
    (tmp_path / "progress.json").write_text(
        json.dumps({"status": "running", "completed_conditions": ["tiny-ct2-int8"]})
    )
    with pytest.raises(ValueError, match="complete 15"):
        complete_summaries(tmp_path)


def test_result_commit_does_not_capture_unrelated_staged_user_work(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.name", "Synthetic Research Test")
    git("config", "user.email", "synthetic@example.test")
    git("config", "commit.gpgsign", "false")
    ours, theirs = tmp_path / "results.json", tmp_path / "user.txt"
    ours.write_text("original result\n")
    theirs.write_text("original user work\n")
    git("add", ".")
    git("commit", "-m", "Synthetic initial fixture")
    ours.write_text("completed result\n")
    theirs.write_text("unrelated staged user change\n")
    git("add", "user.txt")
    publish_owned_paths(tmp_path, [ours], push=False)
    assert git("show", "HEAD:results.json") == "completed result"
    assert git("show", "HEAD:user.txt") == "original user work"
    assert git("diff", "--cached", "--name-only") == "user.txt"


def test_complete_matrix_is_rescored_and_false_cost_is_rejected(tmp_path, monkeypatch):
    rows = [
        {
            "id": str(i),
            "reference": "ọkọ̀ ilé",
            "hypothesis": "ọkọ̀ ilé",
            "duration_seconds": 1.0,
            "processing_seconds": 0.1,
        }
        for i in range(177)
    ]
    checksum = hashlib.sha256(
        json.dumps(
            [{k: row[k] for k in ("id", "reference", "duration_seconds")} for row in rows],
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()
    monkeypatch.setattr("eval.finalize.TEST_EVALUATION_SHA256", checksum)
    labels = [
        size + "-" + variant
        for size in ("tiny", "base", "small")
        for variant in ("ct2-float32", "ct2-int8", "cpp-f16", "cpp-q5_0", "cpp-q4_0")
    ]
    scores = score([r["reference"] for r in rows], [r["hypothesis"] for r in rows])
    for label in labels:
        folder = tmp_path / label
        folder.mkdir()
        (folder / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (folder / "summary.json").write_text(
            json.dumps(
                {
                    "status": "measured",
                    "label": label,
                    "n": 177,
                    "partial": False,
                    "evaluation_set_sha256": checksum,
                    "server_hourly_usd": 0.1,
                    "cost_per_audio_hour_usd": sum(r["processing_seconds"] for r in rows)
                    / 177
                    * 0.1,
                    **scores,
                }
            )
        )
    paired = compare(rows, rows)
    for size in ("tiny", "base", "small"):
        for quant in ("ct2-int8", "cpp-q5_0", "cpp-q4_0"):
            (tmp_path / (size + "-" + quant + "-paired.json")).write_text(json.dumps(paired))
    (tmp_path / "progress.json").write_text(
        json.dumps({"status": "complete", "completed_conditions": labels})
    )
    assert len(complete_summaries(tmp_path)[0]) == 15
    target = tmp_path / labels[0] / "summary.json"
    result = json.loads(target.read_text())
    result["cost_per_audio_hour_usd"] = 0
    target.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="serving cost"):
        complete_summaries(tmp_path)
