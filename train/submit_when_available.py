"""Submit one prepared private bundle after its known predecessor releases a slot."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi

    p = argparse.ArgumentParser()
    p.add_argument("--after", required=True, help="Our preceding owner/kernel/version")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--max-wait-seconds", type=int, default=5400)
    a = p.parse_args()
    api = KaggleApi()
    api.authenticate()
    until = time.monotonic() + a.max_wait_seconds
    while time.monotonic() < until:
        status = str(api.kernels_status(a.after).status).rsplit(".", 1)[-1]
        if status not in {"RUNNING", "QUEUED"}:
            result = api.kernels_push(str(a.bundle), timeout="28800", acc="NvidiaTeslaT4")
            record = {
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "predecessor": a.after,
                "predecessor_status": status,
                "ref": result.ref,
                "version": result.version_number,
                "submitted": not bool(result.error),
            }
            a.record.write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            if result.error:
                raise SystemExit("Kaggle rejected the submission; no automatic repeated pushes")
            return
        time.sleep(30)
    raise SystemExit(
        "Predecessor still active; prepared bundle retained without starting another GPU session"
    )


if __name__ == "__main__":
    main()
