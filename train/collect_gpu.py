"""Wait finitely for a submitted GPU job and retain selected artifacts privately."""

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


def kernel_state(response):
    # The SDK returns a response object with an enum. Its JSON-looking __str__
    # is only a display representation; also accept explicitly serialized forms.
    value = json.loads(response) if isinstance(response, str) else response
    status = value["status"] if isinstance(value, dict) else value.status
    return str(getattr(status, "name", status)).upper()


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi

    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True, help="owner/slug/version")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout-hours", type=float, default=3)
    a = p.parse_args()
    if a.timeout_hours <= 0 or len(a.kernel.split("/")) != 3:
        p.error("Provide an exact kernel version and positive timeout")
    # The SDK also saves raw execution logs; keep those outside the checkout.
    if a.out.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        p.error("Choose a private download directory outside the repository")
    os.umask(0o077)
    a.out.mkdir(parents=True, exist_ok=True, mode=0o700)
    api = KaggleApi()
    api.authenticate()
    deadline = time.monotonic() + 3600 * a.timeout_hours
    while True:
        state = kernel_state(api.kernels_status(a.kernel))
        if state not in {"QUEUED", "RUNNING"}:
            break
        if time.monotonic() >= deadline:
            raise SystemExit("Finite collection wait ended; remote artifacts remain on Kaggle")
        time.sleep(min(60, max(0, deadline - time.monotonic())))
    if shutil.disk_usage(a.out).free < 2 * 1024**3:
        raise SystemExit("Not enough private disk space; remote artifacts remain on Kaggle")
    pattern = (
        r"research-artifacts/(data-audit|job-status)\.json$|"
        r"research-artifacts/(tiny|base|small)/(best/[^/]+|selection\.json|"
        r"test-float/(summary\.json|predictions\.jsonl)|checkpoint-[^/]+/trainer_state\.json)$"
    )
    files, _ = api.kernels_output(a.kernel, str(a.out), file_pattern=pattern, quiet=True)
    result = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "kernel": a.kernel,
        "kernel_status": state,
        "downloaded_files": [str(Path(f).relative_to(a.out)) for f in files],
        "scope": "selected best models, evaluation/selection evidence and private SDK logs; no compute started",
    }
    (a.out / "collection.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Retained", len(files), "files for review; kernel status:", state, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Provider exceptions may contain signed URLs; report only their class.
        raise SystemExit("GPU artifact collection stopped: " + type(error).__name__) from None
