"""Run one bounded CPU experiment, preserve outputs, and destroy its own resources.

Uses the existing scoped AWS profile and SSH key; sends no cloud credentials to
the instance. Terraform state and provisioning logs stay outside the repository.
"""

import argparse
import fcntl
import ipaddress
import json
import os
import re
import shlex
import signal
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from eval.compression import verify_conversion, write_json


def preflight_price(prices, max_hours, budget):
    checked = datetime.fromisoformat(prices["checked_at"])
    age = (datetime.now(timezone.utc) - checked).total_seconds()
    hourly = prices["topologies"]["1xm7i-flex.large"]["cluster_usd_per_hour"]
    if prices["region"] != "us-east-1" or not 0 <= age <= 172800:
        raise ValueError("Refresh the dated us-east-1 public AWS price evidence")
    if not 0 < max_hours <= 48 or not 0 < hourly * max_hours <= budget:
        raise ValueError("Finite compute/storage/IPv4 list-price budget exceeded")
    return hourly


def main():
    import boto3

    p = argparse.ArgumentParser()
    p.add_argument("--account-id", required=True)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--prices", type=Path, required=True)
    p.add_argument("--terraform", type=Path, required=True)
    p.add_argument("--plugin-dir", type=Path)
    p.add_argument("--ssh-key", type=Path, default=Path.home() / ".ssh/afri-ai")
    p.add_argument("--max-hours", type=float, default=24)
    p.add_argument("--budget-usd", type=float, default=6)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if not a.out.resolve().is_relative_to(repo / "report/runs"):
        p.error("Keep benchmark evidence under this repository's report/runs/")
    if not re.fullmatch(r"[0-9]{12}", a.account_id):
        p.error("Supply the intended AWS account")
    prices = json.loads(a.prices.read_text())
    hourly = preflight_price(prices, a.max_hours, a.budget_usd)
    a.converted = a.converted.resolve(strict=True)
    a.manifest = a.manifest.resolve(strict=True)
    a.terraform = a.terraform.resolve(strict=True)
    a.ssh_key = a.ssh_key.expanduser().resolve(strict=True)
    verify_conversion(a.converted)
    rows = [json.loads(line) for line in a.manifest.read_text().splitlines()]
    if len(rows) != 177 or len({Path(r["audio"]).name for r in rows}) != 177:
        raise ValueError("Use all 177 distinct prepared recordings")
    audio_dirs = {Path(r["audio"]).resolve(strict=True).parent for r in rows}
    if len(audio_dirs) != 1:
        raise ValueError("Stage the reviewed test audio in a single prepared directory")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            "HEAD",
            "--",
            "eval",
            "serve",
            "train/model_revisions.py",
            "requirements.lock",
        ],
        cwd=repo,
        text=True,
    ).strip():
        raise ValueError("Commit and push benchmark sources before provisioning")
    session = boto3.Session(profile_name="afri-research", region_name="us-east-1")
    if session.client("sts").get_caller_identity()["Account"] != a.account_id:
        raise ValueError("Unexpected AWS account")
    state = Path.home() / ".local/state/afri-asr-compression"
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.umask(0o077)
    lock = (state / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    a.out.mkdir(parents=True, exist_ok=True)
    a.out = a.out.resolve()
    env = {**os.environ, "TF_DATA_DIR": str(state / "terraform"), "TF_IN_AUTOMATION": "1"}
    tf = [str(a.terraform), "-chdir=" + str(repo / "serve/infra")]
    receipt = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": revision,
        "status": "preparing",
        "hourly_usd": hourly,
        "max_hours": a.max_hours,
        "maximum_compute_storage_ipv4_usd": hourly * a.max_hours,
        "cost_exclusions": "Tax and traffic; credit balance not independently accessible",
        "stages": [],
    }
    host = None
    expected_instance = None
    applied = False
    deadline = time.monotonic() + a.max_hours * 3600

    def interrupted(*_):
        raise SystemExit("Controller interrupted; collecting outputs and tearing down")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def run(name, command, timeout=600, *, input=None, check=True):
        with (state / (name + ".log")).open("a") as log:
            result = subprocess.run(
                command,
                cwd=repo,
                env=env,
                text=True,
                input=input,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        receipt["stages"].append(
            {
                "name": name,
                "exit_code": result.returncode,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(a.out / "controller.json", receipt)
        if check and result.returncode:
            raise RuntimeError("Benchmark stage failed: " + name)
        return result.returncode

    def outputs():
        values = json.loads(
            subprocess.check_output(tf + ["output", "-json"], cwd=repo, env=env, text=True)
        )
        return {key: value["value"] for key, value in values.items()}

    ssh = [
        "ssh",
        "-i",
        str(a.ssh_key),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "UserKnownHostsFile=" + str(state / "known_hosts"),
    ]

    def remote(command, timeout=600, **kwargs):
        return run("remote-bootstrap", ssh + ["ubuntu@" + host, command], timeout, **kwargs)

    def copy(source, destination, *, check=True):
        return run(
            "artifact-transfer",
            ["rsync", "-aL", "--timeout=90", "-e", shlex.join(ssh), str(source), destination],
            1800,
            check=check,
        )

    def collect():
        if host:
            copy("ubuntu@" + host + ":/home/ubuntu/asr-results/", str(a.out) + "/", check=False)

    try:
        init = tf + [
            "init",
            "-input=false",
            "-reconfigure",
            "-no-color",
            "-backend-config=path=" + str(state / "terraform.tfstate"),
        ]
        if a.plugin_dir:
            init += ["-plugin-dir=" + str(a.plugin_dir.resolve())]
        run("terraform-init", init)
        if subprocess.check_output(tf + ["state", "list"], env=env, cwd=repo, text=True).strip():
            raise ValueError("Dedicated benchmark state is not empty; inspect the previous run")
        request = urllib.request.Request(
            "https://checkip.amazonaws.com", headers={"User-Agent": "afri-asr-research"}
        )
        address = str(
            ipaddress.IPv4Address(
                urllib.request.urlopen(request, timeout=15).read().decode().strip()
            )
        )
        variables = {
            "aws_account_id": a.account_id,
            "admin_cidr": address + "/32",
            "ssh_public_key_path": str(a.ssh_key) + ".pub",
            "max_hours": a.max_hours,
        }
        varfile = state / "benchmark.tfvars.json"
        write_json(varfile, variables)
        applied = True  # A failed apply may still have created resources.
        run(
            "terraform-apply",
            tf
            + ["apply", "-auto-approve", "-input=false", "-no-color", "-var-file=" + str(varfile)],
            900,
        )
        inventory = outputs()
        expected_instance = inventory["instance_id"]
        host = str(ipaddress.IPv4Address(inventory["public_ip"]))
        receipt["resources"] = inventory
        receipt["status"] = "provisioned"
        write_json(a.out / "controller.json", receipt)
        ec2 = session.client("ec2")
        instance = ec2.describe_instances(InstanceIds=[expected_instance])["Reservations"][0][
            "Instances"
        ][0]
        receipt["launch_time"] = instance["LaunchTime"].isoformat()
        receipt["volume_ids"] = [v["Ebs"]["VolumeId"] for v in instance["BlockDeviceMappings"]]
        for _ in range(60):
            if remote("true", timeout=15, check=False) == 0:
                break
            time.sleep(5)
        else:
            raise TimeoutError("New instance SSH did not become ready")
        remote("sudo cloud-init status --wait", timeout=900)
        remote("systemctl is-active afri-benchmark-expiry.timer")
        receipt["instance_expiry_timer_verified"] = True
        bootstrap = f"""set -eu
git clone --quiet https://github.com/Eunini/yoruba-asr-tone-compression.git /home/ubuntu/study
git -C /home/ubuntu/study checkout --quiet --detach {shlex.quote(revision)}
python3 -m venv /home/ubuntu/asr-venv
/home/ubuntu/asr-venv/bin/pip install --disable-pip-version-check -r /home/ubuntu/study/requirements.lock
git init -q /home/ubuntu/whisper.cpp
git -C /home/ubuntu/whisper.cpp remote add origin https://github.com/ggml-org/whisper.cpp.git
git -C /home/ubuntu/whisper.cpp fetch --depth=1 origin 231638a0717b7822784297e5980965ffcec05810
git -C /home/ubuntu/whisper.cpp checkout --detach FETCH_HEAD
cmake -S /home/ubuntu/whisper.cpp -B /home/ubuntu/whisper.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=OFF
cmake --build /home/ubuntu/whisper.cpp/build --target whisper-server -j2
mkdir -p /home/ubuntu/asr-artifacts/converted /home/ubuntu/asr-artifacts/test /home/ubuntu/asr-results
lscpu --json > /home/ubuntu/asr-results/lscpu.json
uname -a > /home/ubuntu/asr-results/kernel.txt
/home/ubuntu/asr-venv/bin/pip freeze > /home/ubuntu/asr-results/packages.txt
"""
        remote("bash -s", input=bootstrap, timeout=2400)
        copy(str(a.converted) + "/", "ubuntu@" + host + ":/home/ubuntu/asr-artifacts/converted/")
        copy(
            str(next(iter(audio_dirs))) + "/",
            "ubuntu@" + host + ":/home/ubuntu/asr-artifacts/test/",
        )
        remote_manifest = state / "test.jsonl"
        remote_manifest.write_text(
            "".join(
                json.dumps(
                    {**r, "audio": "/home/ubuntu/asr-artifacts/test/" + Path(r["audio"]).name},
                    ensure_ascii=False,
                )
                + "\n"
                for r in rows
            )
        )
        copy(remote_manifest, "ubuntu@" + host + ":/home/ubuntu/asr-artifacts/test.jsonl")
        remaining = (deadline - time.monotonic()) / 3600 - 0.25
        if remaining < 0.25:
            raise TimeoutError("Insufficient bounded runtime remains after setup")
        command = [
            "sudo",
            "systemd-run",
            "--unit=afri-asr-matrix",
            "--uid=ubuntu",
            "--working-directory=/home/ubuntu/study",
            "--property=CPUQuota=200%",
            "--property=MemoryMax=6G",
            f"--property=RuntimeMaxSec={int(remaining * 3600)}",
            "--property=StandardOutput=append:/home/ubuntu/asr-results/matrix.log",
            "--property=StandardError=append:/home/ubuntu/asr-results/matrix.log",
            "/home/ubuntu/asr-venv/bin/python",
            "-m",
            "eval.compression",
            "--converted",
            "/home/ubuntu/asr-artifacts/converted",
            "--manifest",
            "/home/ubuntu/asr-artifacts/test.jsonl",
            "--cpp-server",
            "/home/ubuntu/whisper.cpp/build/bin/whisper-server",
            "--out",
            "/home/ubuntu/asr-results",
            "--hardware-label",
            "AWS us-east-1 m7i-flex.large; dedicated 2 vCPU 8 GiB; variable Flex performance",
            "--threads",
            "2",
            "--max-hours",
            str(remaining),
            "--server-hourly-usd",
            str(hourly),
            "--price-source",
            prices["sources"][0]["url"],
        ]
        remote(shlex.join(command))
        receipt["status"] = "measuring"
        write_json(a.out / "controller.json", receipt)
        print(
            "Dedicated CPU matrix started; outputs sync every minute and resources have a finite expiry",
            flush=True,
        )
        while time.monotonic() < deadline - 120:
            collect()
            progress_path = a.out / "progress.json"
            if progress_path.exists():
                progress = json.loads(progress_path.read_text())
                if progress.get("status") in {"complete", "stopped"}:
                    receipt["matrix_status"] = progress
                    if progress["status"] != "complete":
                        raise RuntimeError("CPU matrix stopped; partial results retained")
                    receipt["status"] = "measurements_complete"
                    break
            status = subprocess.run(
                ssh + ["ubuntu@" + host, "systemctl is-active afri-asr-matrix"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if status.returncode and status.stdout.strip() in {"inactive", "failed"}:
                raise RuntimeError("Remote matrix service stopped; inspect synced log")
            time.sleep(60)
        else:
            raise TimeoutError("Finite controller runtime reached")
    except BaseException as error:
        receipt.update(status="stopped", error_type=type(error).__name__)
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if host:
            try:
                collect()
            except (OSError, subprocess.SubprocessError):
                receipt["final_sync_failed"] = True
        if applied:
            try:
                current = outputs()
            except (subprocess.SubprocessError, json.JSONDecodeError):
                if expected_instance:
                    raise RuntimeError(
                        "Cannot re-check the owned instance before teardown"
                    ) from None
                current = {}  # A partial apply can have resources but no outputs.
            if expected_instance and current.get("instance_id") != expected_instance:
                raise RuntimeError("Instance set changed; refusing to destroy a different run")
            code = run(
                "terraform-destroy",
                tf
                + [
                    "destroy",
                    "-auto-approve",
                    "-input=false",
                    "-no-color",
                    "-var-file=" + str(state / "benchmark.tfvars.json"),
                ],
                900,
                check=False,
            )
            receipt["terraform_destroy_exit_code"] = code
            empty = not subprocess.check_output(
                tf + ["state", "list"], cwd=repo, env=env, text=True
            ).strip()
            receipt["terraform_state_empty"] = empty
            if expected_instance:
                instances = session.client("ec2").describe_instances(
                    InstanceIds=[expected_instance]
                )["Reservations"]
                receipt["instance_terminated"] = all(
                    i["State"]["Name"] == "terminated" for r in instances for i in r["Instances"]
                )
            receipt["teardown_verified"] = (
                code == 0 and empty and receipt.get("instance_terminated", True)
            )
            if not receipt["teardown_verified"]:
                write_json(a.out / "controller.json", receipt)
                raise RuntimeError("Teardown not verified; inspect the exclusive benchmark state")
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(a.out / "controller.json", receipt)


if __name__ == "__main__":
    main()
