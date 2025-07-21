"""Prepare a private Kaggle code dataset and GPU notebook from a committed tree."""

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--owner", required=True, help="Kaggle username, not a token")
    p.add_argument("--out", type=Path, required=True, help="Bundle directory outside git")
    p.add_argument("--skip-baselines", action="store_true", help="Retain a separately running baseline stage; train and test only")
    a = p.parse_args()
    dataset = a.out / "dataset"
    kernel = a.out / "kernel"
    dataset.mkdir(parents=True, exist_ok=True)
    kernel.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(["git", "archive", "--format=tar.gz", "HEAD"])
    (dataset / "code.tar.gz").write_bytes(archive)
    (dataset / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Yoruba ASR tone compression research code",
                "id": a.owner + "/yoruba-asr-tone-code",
                "licenses": [{"name": "other"}],
                "description": "Private source-code snapshot. The archive includes its MIT code license and dataset attribution; no source audio or credentials are bundled.",
            },
            indent=2,
        )
    )
    bootstrap = """import os, sys, tarfile, pathlib, subprocess, base64, io, hashlib, urllib.request
os.environ['HF_HOME'] = '/kaggle/temp/huggingface'
root = pathlib.Path('/kaggle/temp/yoruba-asr-tone-compression')
root.mkdir(parents=True, exist_ok=True)
blob = base64.b64decode('__CODE_B64__')
assert hashlib.sha256(blob).hexdigest() == '__CODE_SHA__', 'Source checksum mismatch'
with tarfile.open(fileobj=io.BytesIO(blob)) as source:
    source.extractall(root, filter='data')
import torch
print('Python:', sys.version, 'Torch:', torch.__version__, 'CUDA build:', torch.version.cuda)
assert torch.cuda.is_available(), 'Kaggle did not provide a usable CUDA runtime; no training started'
print('GPU:', torch.cuda.get_device_name(0), 'count:', torch.cuda.device_count())
with urllib.request.urlopen('https://pypi.org/pypi/jiwer/json', timeout=30) as response:
    print('Package network check:', response.status)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-build-isolation',
                '--index-url', 'https://pypi.org/simple', '-e', str(root) + '[training]'], check=True)
os.chdir(root)
subprocess.run([sys.executable, '-m', 'data.prepare'], check=True)
"""
    bootstrap = bootstrap.replace("__CODE_B64__", base64.b64encode(archive).decode())
    bootstrap = bootstrap.replace("__CODE_SHA__", hashlib.sha256(archive).hexdigest())
    study = """import shutil, json
output = pathlib.Path('/kaggle/working/research-artifacts')
output.mkdir(exist_ok=True)
shutil.copy('data/prepared/audit.json', output / 'data-audit.json')
jobs = []
def run_job(name, command):
    environment = os.environ.copy()
    # Single-GPU Trainer avoids DataParallel/checkpoint graph reuse on the
    # current Kaggle Torch build. Effective batch: 4 x accumulation 4 = 16.
    environment['CUDA_VISIBLE_DEVICES'] = '0'
    status = subprocess.run(command, env=environment).returncode
    jobs.append({'job':name, 'return_code':status})
    (output/'job-status.json').write_text(json.dumps(jobs,indent=2))
    return status == 0
# Independent stages retain their artifacts even if another stage fails.
for size in ['tiny', 'base', 'small']:
    destination = output / size
    if run_job('train-'+size, [sys.executable, '-m', 'train.finetune', '--size', size,
                    '--out', str(destination), '--steps', '2000']):
        run_job('test-'+size, [sys.executable, '-m', 'eval.run', '--manifest', 'data/prepared/test.jsonl',
                    '--backend', 'whisper', '--model', str(destination / 'best'),
                    '--device', 'cuda', '--out', str(destination / 'test-float')])
if __RUN_BASELINES__:
    for backend, model in [('whisper', 'openai/whisper-small'), ('mms', 'facebook/mms-1b-all')]:
        run_job('baseline-'+backend, [sys.executable, '-m', 'eval.run', '--manifest', 'data/prepared/test.jsonl',
                        '--backend', backend, '--model', model, '--device', 'cuda',
                        '--out', str(output / ('baseline-' + backend))])
print('Selected models and GPU evaluations are in', output)
print('Compression, controlled CPU costs, and native error review remain separate stages.')
if any(job['return_code'] for job in jobs):
    raise RuntimeError('Some stages failed; job-status.json records them. Successful artifacts are retained.')
"""
    study = study.replace("__RUN_BASELINES__", str(not a.skip_baselines))
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Yoruba tone/compression: GPU training stage\n",
                "Private notebook. Public source datasets are downloaded without embedding any account token.\n",
                "Run baselines and validation-selected training. CPU compression/cost experiments follow.\n",
            ],
        },
        *[
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": text.splitlines(keepends=True),
            }
            for text in [bootstrap, study]
        ],
    ]
    for i, cell in enumerate(cells):
        cell["id"] = f"research-cell-{i}"
    (kernel / "study.ipynb").write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3",
                    }
                },
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            indent=2,
        )
    )
    (kernel / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": a.owner + "/yoruba-asr-tone-compression-gpu-study",
                "title": "Yoruba ASR tone compression GPU study",
                "code_file": "study.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4",
                "docker_image": "gcr.io/kaggle-private-byod/python@sha256:22be155a37fc52bc8254ebfd0bb894643d00a1283e2446e212597ae12aec45f9",
                "docker_image_pinning_type": "original",
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": [],
            },
            indent=2,
        )
    )
    print(
        "Prepared private dataset and kernel bundles. Use Kaggle CLI create/push after authentication."
    )


if __name__ == "__main__":
    main()
