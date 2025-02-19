"""Prepare a private Kaggle code dataset and GPU notebook from a committed tree."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--owner", required=True, help="Kaggle username, not a token")
    p.add_argument("--out", type=Path, required=True, help="Bundle directory outside git")
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
                "licenses": [{"name": "copyright-authors"}],
                "description": "Private source-code snapshot. The archive includes its MIT code license and dataset attribution; no source audio or credentials are bundled.",
            },
            indent=2,
        )
    )
    bootstrap = """import os, sys, glob, tarfile, pathlib, subprocess
os.environ['HF_HOME'] = '/kaggle/temp/huggingface'
root = pathlib.Path('/kaggle/temp/yoruba-asr-tone-compression')
root.mkdir(parents=True, exist_ok=True)
archive = glob.glob('/kaggle/input/**/code.tar.gz', recursive=True)
assert len(archive) == 1, 'Attach the private code dataset'
with tarfile.open(archive[0]) as source:
    source.extractall(root, filter='data')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', str(root) + '[training,inference]'], check=True)
os.chdir(root)
import torch
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator'
print('GPU:', torch.cuda.get_device_name(0))
subprocess.run([sys.executable, '-m', 'data.prepare'], check=True)
"""
    study = """import shutil
output = pathlib.Path('/kaggle/working/research-artifacts')
output.mkdir(exist_ok=True)
shutil.copy('data/prepared/audit.json', output / 'data-audit.json')
for backend, model in [('whisper', 'openai/whisper-small'), ('mms', 'facebook/mms-1b-all')]:
    subprocess.run([sys.executable, '-m', 'eval.run', '--manifest', 'data/prepared/test.jsonl',
                    '--backend', backend, '--model', model, '--device', 'cuda',
                    '--out', str(output / ('baseline-' + backend))], check=True)
for size in ['tiny', 'base', 'small']:
    destination = output / size
    subprocess.run([sys.executable, '-m', 'train.finetune', '--size', size,
                    '--out', str(destination), '--steps', '2000'], check=True)
    subprocess.run([sys.executable, '-m', 'eval.run', '--manifest', 'data/prepared/test.jsonl',
                    '--backend', 'whisper', '--model', str(destination / 'best'),
                    '--device', 'cuda', '--out', str(destination / 'test-float')], check=True)
print('Selected models and GPU evaluations are in', output)
print('Compression, controlled CPU costs, and native error review remain separate stages.')
"""
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
                "id": a.owner + "/yoruba-asr-tone-compression",
                "title": "Yoruba ASR tone compression GPU study",
                "code_file": "study.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [a.owner + "/yoruba-asr-tone-code"],
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
