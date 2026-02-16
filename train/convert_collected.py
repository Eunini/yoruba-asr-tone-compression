"""Convert only fingerprint-verified checkpoints after a finite collection wait."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from eval.artifacts import fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--collected", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--converters", type=Path, required=True)
    p.add_argument("--wait-seconds", type=int, default=10800)
    a = p.parse_args()
    if a.wait_seconds < 0:
        p.error("Use a nonnegative finite collection wait")
    deadline = time.monotonic() + a.wait_seconds
    while not (a.collected / "collection.json").is_file():
        if time.monotonic() >= deadline:
            raise SystemExit("Collection has not finished; no conversion started")
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        PATH=str(Path(sys.executable).parent) + os.pathsep + env["PATH"],
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
    )
    a.out.mkdir(parents=True, exist_ok=True)
    converted, missing = [], []
    for size in ("tiny", "base", "small"):
        source = a.collected / "research-artifacts" / size
        if not (source / "selection.json").is_file() or not (
            source / "test-float/summary.json"
        ).is_file():
            missing.append(size)
            continue
        selection = json.loads((source / "selection.json").read_text())
        evaluation = json.loads((source / "test-float/summary.json").read_text())
        best = source / "best"
        expected = selection.get("artifacts", {})
        actual = fingerprint(best)
        if (
            not expected
            or selection.get("base_model") != "openai/whisper-" + size
            or evaluation.get("status") != "measured"
            or evaluation.get("partial")
            or evaluation.get("n") != 177
            or any(actual.get(k) != v for k, v in expected.items())
            or any(evaluation.get("artifacts", {}).get(k) != v for k, v in expected.items())
        ):
            raise ValueError("Checkpoint selection/evaluation fingerprint mismatch: " + size)
        target = a.out / size
        if target.exists():
            raise FileExistsError("Inspect existing conversion output before repeating: " + size)
        if shutil.disk_usage(a.out).free < 6 * 1024**3:
            raise RuntimeError("Insufficient conversion scratch space; checkpoints preserved")
        subprocess.run(
            [
                "bash", str(repo / "scripts/compress.sh"), str(best), str(target),
                str(a.converters / "whisper.cpp"), str(a.converters / "whisper"),
            ],
            cwd=repo, env=env, check=True, timeout=3600,
        )
        expected_files = [
            target / "ct2-int8/model.bin", target / "ct2-float32/model.bin",
            target / "ggml-model.bin", target / "ggml-model-q5_0.bin",
            target / "ggml-model-q4_0.bin",
        ]
        if any(not f.is_file() or not f.stat().st_size for f in expected_files):
            raise ValueError("A converter returned without all expected artifacts")
        # This export is reproducible intermediate data; selected source weights
        # remain immutable in private persistent storage.
        shutil.rmtree(target / "hf-bin")
        outputs = fingerprint(target)
        for backend in ("ct2-int8", "ct2-float32"):
            outputs.update({backend + "/" + k: v for k, v in fingerprint(target / backend).items()})
        record = {
            "status": "converted; inference validation pending",
            "source_artifacts": expected,
            "converted_artifacts": outputs,
            "toolchain": json.loads((repo / "report/conversion-toolchain.json").read_text()),
        }
        (target / "conversion.json").write_text(json.dumps(record, indent=2) + "\n")
        converted.append(size)
        print("Converted and fingerprinted:", size, flush=True)
    (a.out / "conversion-status.json").write_text(
        json.dumps({"converted_sizes": converted, "missing_selected_evaluations": missing}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
