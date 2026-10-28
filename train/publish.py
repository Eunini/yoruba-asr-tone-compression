"""Publish an actually selected fine-tuned model with measured evaluation evidence."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from eval.artifacts import fingerprint


TEST_SET = "2e71d67cf2d5de5c1953bc646ca1550887b31de11f2b6370e82beacaf9cc8529"


def prepare_release(checkpoint_root, evaluation, card, notices, staging, model_license):
    """Build an upload package without changing the selected checkpoint."""
    selection = json.loads((checkpoint_root / "selection.json").read_text())
    result = json.loads(evaluation.read_text())
    best = checkpoint_root / "best"
    if (
        result.get("status") != "measured"
        or result.get("partial")
        or not selection.get("checkpoint")
        or result.get("n") != 177
        or result.get("evaluation_set_sha256") != TEST_SET
    ):
        raise ValueError("A real selected checkpoint and complete measured evaluation are required")
    if not any(best.glob("*.safetensors")):
        raise ValueError("Publish safetensors weights; pickle training state is excluded")
    expected = selection.get("artifacts", {})
    actual = fingerprint(best)
    if not expected or actual != expected:
        raise ValueError("Local weights/configuration differ from the selected checkpoint")
    if any(result.get("artifacts", {}).get(name) != value for name, value in expected.items()):
        raise ValueError("Evaluation does not fingerprint the selected trained weights")
    # Parse the reviewed card through the same lightweight metadata parser as HF.
    from huggingface_hub import ModelCard

    metadata = ModelCard(card.read_text()).data.to_dict()
    if (
        metadata.get("license") != model_license
        or metadata.get("base_model") != selection["base_model"]
    ):
        raise ValueError("Reviewed card license/base model disagrees with this release")
    required_notices = {
        "LICENSE",
        "LICENSE-BASE-APACHE-2.0.txt",
        "LICENSE-UPSTREAM-WHISPER-MIT.txt",
        "NOTICE.md",
    }
    if not required_notices.issubset({p.name for p in notices.iterdir() if p.is_file()}):
        raise ValueError("Retain the adaptation license, upstream licenses and attribution notice")
    if staging.resolve().is_relative_to(checkpoint_root.resolve()):
        raise ValueError("Stage the release outside the immutable checkpoint")
    staging.mkdir(parents=True, exist_ok=False)
    included = []
    for name in sorted(expected):
        if Path(name).name != name:
            raise ValueError("Unexpected nested checkpoint artifact")
        if Path(name).suffix in {".safetensors", ".json", ".txt"}:
            shutil.copy2(best / name, staging / name)
            included.append(name)
    shutil.copy2(card, staging / "README.md")
    for name in sorted(required_notices):
        shutil.copy2(notices / name, staging / name)
    shutil.copy2(checkpoint_root / "selection.json", staging / "selection.json")
    shutil.copy2(evaluation, staging / "evaluation.json")
    manifest = {
        "base_model": selection["base_model"],
        "base_model_revision": selection["base_model_revision"],
        "selected_checkpoint": selection["checkpoint"],
        "source_artifacts": expected,
        "inference_artifacts_included": included,
        "artifacts_excluded": sorted(set(expected) - set(included)),
        "model_license": model_license,
        "files": {p.name: fingerprint(p)[p.name] for p in sorted(staging.iterdir())},
    }
    (staging / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if fingerprint(best) != expected:
        raise ValueError("Selected checkpoint changed while preparing release")
    return manifest


def main():
    from huggingface_hub import HfApi

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--card", type=Path, required=True)
    p.add_argument("--notices", type=Path, required=True)
    p.add_argument("--staging", type=Path, required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--model-license", required=True)
    p.add_argument("--publish", action="store_true", help="Without this flag, prepare/verify only")
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "train/publish.py",
            "eval/artifacts.py",
            "report/model-release",
        ],
        cwd=repo,
        text=True,
    ).strip():
        raise ValueError("Commit the reviewed publication sources and model card first")
    manifest = prepare_release(
        a.checkpoint_root, a.evaluation, a.card, a.notices, a.staging, a.model_license
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    manifest["publication_source_commit"] = revision
    (a.staging / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not a.publish:
        print(
            json.dumps(
                {
                    "status": "prepared; no upload",
                    "staging": str(a.staging),
                    "files": len(manifest["files"]),
                }
            )
        )
        return
    token_path = Path.home() / ".cache/huggingface/token"
    token = os.environ.get("HF_TOKEN") or token_path.read_text().strip()
    api = HfApi(token=token)
    if api.whoami()["name"] != a.repo_id.split("/")[0]:
        raise ValueError("Unexpected Hugging Face publishing identity")
    # A fresh name prevents accidentally modifying an unrelated existing model.
    api.create_repo(a.repo_id, exist_ok=False, private=False)
    commit = api.upload_folder(
        repo_id=a.repo_id,
        folder_path=a.staging,
        allow_patterns=list(manifest["files"]) + ["release-manifest.json"],
        commit_message="Publish selected Yoruba ASR checkpoint with measured evaluation",
    )
    print(
        json.dumps(
            {
                "status": "uploaded",
                "url": "https://huggingface.co/" + a.repo_id,
                "commit": commit.oid,
            }
        )
    )


if __name__ == "__main__":
    main()
