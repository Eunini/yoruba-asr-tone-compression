"""Bounded, resumable full-test matrix with persistent CPU models and paired TER."""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from eval.paired import compare

TEST_EVALUATION_SHA256 = "2e71d67cf2d5de5c1953bc646ca1550887b31de11f2b6370e82beacaf9cc8529"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_conversion(root):
    for size in ("tiny", "base", "small"):
        folder = root / size
        receipt = json.loads((folder / "conversion.json").read_text())
        for name, expected in receipt["converted_artifacts"].items():
            if digest(folder / name) != expected:
                raise ValueError("Converted artifact fingerprint mismatch: " + size + "/" + name)


def main():
    import httpx

    p = argparse.ArgumentParser()
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cpp-server", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hardware-label", required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--port", type=int, default=18992)
    p.add_argument("--max-hours", type=float, default=24)
    p.add_argument("--server-hourly-usd", type=float)
    p.add_argument("--price-source")
    a = p.parse_args()
    if not 0 < a.max_hours <= 48 or a.threads < 1:
        p.error("Use positive threads and a finite runtime of at most 48 hours")
    if a.server_hourly_usd is not None and (a.server_hourly_usd < 0 or not a.price_source):
        p.error("A measured-cost run needs a public price-source URL")
    a.converted = a.converted.resolve(strict=True)
    a.cpp_server = a.cpp_server.resolve(strict=True)
    a.manifest = a.manifest.resolve(strict=True)
    rows = [json.loads(line) for line in a.manifest.read_text().splitlines()]
    if len(rows) != 177 or len({r["id"] for r in rows}) != 177:
        raise ValueError("Use all 177 distinct held-out WAXAL recordings")
    canonical = hashlib.sha256(
        json.dumps(
            [{key: row[key] for key in ("id", "reference", "duration_seconds")} for row in rows],
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if canonical != TEST_EVALUATION_SHA256:
        raise ValueError("Test content or order differs from the audited corpus")
    if any(not Path(r["audio"]).is_file() for r in rows):
        raise ValueError("Manifest audio is missing")
    a.out.mkdir(parents=True, exist_ok=True)
    lock = (a.out / ".matrix.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_conversion(a.converted)
    identity = {
        "manifest_sha256": digest(a.manifest),
        "cpp_server_sha256": digest(a.cpp_server),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "matrix_source_sha256": digest(Path(__file__)),
        "conversion_receipts": {
            size: digest(a.converted / size / "conversion.json")
            for size in ("tiny", "base", "small")
        },
        "hardware_label": a.hardware_label,
        "platform": platform.platform(),
        "threads": a.threads,
        "server_hourly_usd": a.server_hourly_usd,
        "price_source": a.price_source,
        "port": a.port,
        "scope": "15 converted full-test CPU conditions; 9 within-backend paired comparisons; original HF GPU runs are separate",
    }
    identity_path = a.out / "matrix-identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Matrix identity changed; use a separate output directory")
    if not identity_path.exists():
        write_json(identity_path, identity)
    progress = {
        "status": "running",
        "completed_conditions": [],
        "total_conditions": 15,
        "deadline_hours": a.max_hours,
    }
    deadline = time.monotonic() + a.max_hours * 3600

    def interrupted(*_):
        raise SystemExit("Matrix interrupted; saved predictions retained")

    signal.signal(signal.SIGTERM, interrupted)
    env = {
        **os.environ,
        "OMP_NUM_THREADS": str(a.threads),
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": str(a.threads),
        "TOKENIZERS_PARALLELISM": "false",
    }
    try:
        write_json(a.out / "progress.json", progress)
        for size in ("tiny", "base", "small"):
            folder = a.converted / size
            for label, backend, artifact, precision in [
                ("ct2-float32", "ct2", folder / "ct2-float32", "float32"),
                ("ct2-int8", "ct2", folder / "ct2-int8", "int8"),
                ("cpp-f16", "cpp", folder / "ggml-model.bin", "float32"),
                ("cpp-q5_0", "cpp", folder / "ggml-model-q5_0.bin", "float32"),
                ("cpp-q4_0", "cpp", folder / "ggml-model-q4_0.bin", "float32"),
            ]:
                name = size + "-" + label
                target = a.out / name
                command = [
                    sys.executable,
                    "-m",
                    "eval.run",
                    "--manifest",
                    str(a.manifest),
                    "--backend",
                    backend,
                    "--model",
                    str(artifact),
                    "--label",
                    name,
                    "--compute-type",
                    precision,
                    "--threads",
                    str(a.threads),
                    "--out",
                    str(target),
                    "--resume",
                ]
                if a.server_hourly_usd is not None:
                    command += [
                        "--server-hourly-usd",
                        str(a.server_hourly_usd),
                        "--price-source",
                        a.price_source,
                    ]
                server = None
                with (a.out / (name + ".log")).open("a") as log:
                    try:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("Finite matrix runtime reached")
                        if backend == "cpp":
                            with socket.socket() as port_check:
                                port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                                port_check.bind(("127.0.0.1", a.port))
                            server_command = [
                                str(a.cpp_server),
                                "--model",
                                str(artifact),
                                "--threads",
                                str(a.threads),
                                "--host",
                                "127.0.0.1",
                                "--port",
                                str(a.port),
                                "--no-gpu",
                                "--language",
                                "yo",
                                "--best-of",
                                "1",
                                "--beam-size",
                                "1",
                                "--no-fallback",
                            ]
                            server_start = time.perf_counter()
                            server = subprocess.Popen(
                                server_command, env=env, stdout=log, stderr=subprocess.STDOUT
                            )
                            endpoint = f"http://127.0.0.1:{a.port}"
                            with httpx.Client(timeout=2) as client:
                                for _ in range(180):
                                    if server.poll() is not None:
                                        raise RuntimeError(
                                            "whisper.cpp server exited; inspect condition log"
                                        )
                                    try:
                                        if client.get(endpoint).status_code == 200:
                                            break
                                    except httpx.HTTPError:
                                        pass
                                    time.sleep(1)
                                else:
                                    raise TimeoutError("whisper.cpp model readiness timed out")
                            target.mkdir(parents=True, exist_ok=True)
                            with (target / "server-load-events.jsonl").open("a") as events:
                                events.write(
                                    json.dumps(
                                        {
                                            "server_startup_seconds": time.perf_counter()
                                            - server_start,
                                            "timestamp": datetime.now(timezone.utc).isoformat(),
                                            "server_command": server_command,
                                        }
                                    )
                                    + "\n"
                                )
                            command += ["--cpp-server-url", endpoint]
                        subprocess.run(
                            command,
                            env=env,
                            check=True,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=max(1, deadline - time.monotonic()),
                        )
                    finally:
                        if server is not None:
                            server.terminate()
                            try:
                                server.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                server.kill()
                                server.wait()
                summary = json.loads((target / "summary.json").read_text())
                if summary.get("n") != 177 or summary.get("partial"):
                    raise ValueError("A matrix condition did not complete the full test")
                progress["completed_conditions"].append(name)
                write_json(a.out / "progress.json", progress)
                print("Completed full-test condition:", name, flush=True)
            for control, quantized in [
                ("ct2-float32", "ct2-int8"),
                ("cpp-f16", "cpp-q5_0"),
                ("cpp-f16", "cpp-q4_0"),
            ]:

                def read(label):
                    return [
                        json.loads(line)
                        for line in (a.out / (size + "-" + label) / "predictions.jsonl")
                        .read_text()
                        .splitlines()
                    ]

                result = compare(read(control), read(quantized))
                result.update(size=size, control=control, quantized=quantized)
                write_json(a.out / (size + "-" + quantized + "-paired.json"), result)
        progress["status"] = "complete"
    except BaseException as error:
        progress.update(status="stopped", error_type=type(error).__name__)
        raise
    finally:
        progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(a.out / "progress.json", progress)


if __name__ == "__main__":
    main()
