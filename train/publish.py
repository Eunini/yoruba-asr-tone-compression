"""Publish an actually selected fine-tuned model with measured evaluation evidence."""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi
from eval.artifacts import fingerprint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument(
        "--model-license", required=True, help="Reviewed Hugging Face license identifier"
    )
    a = p.parse_args()
    selection = json.loads((a.checkpoint_root / "selection.json").read_text())
    result = json.loads(a.evaluation.read_text())
    best = a.checkpoint_root / "best"
    if (
        result.get("status") != "measured"
        or result.get("partial")
        or not selection.get("checkpoint")
    ):
        raise ValueError("A real selected checkpoint and complete measured evaluation are required")
    if not any(best.glob("*.safetensors")) and not (best / "pytorch_model.bin").exists():
        raise ValueError("No trained weights found")
    # Evaluation must identify these trained weights, not a pretrained baseline.
    expected = selection.get("artifacts", {})
    actual = fingerprint(best)
    if not expected or any(actual.get(name) != value for name, value in expected.items()):
        raise ValueError("Local weights/configuration differ from the selected checkpoint")
    if any(result.get("artifacts", {}).get(name) != value for name, value in expected.items()):
        raise ValueError("Evaluation does not fingerprint the selected trained weights")
    card = f"""---
language: [yo]
license: {a.model_license}
base_model: openai/whisper-{best.parent.name}
datasets: [google/WaxalNLP]
pipeline_tag: automatic-speech-recognition
---
# Yoruba speech recognition — measured research checkpoint

Fine-tuned on the WAXAL Yoruba TTS recordings repurposed for ASR.
The source Media Trust data uses CC-BY-SA-4.0; derivative-model licensing
requirements should be reviewed before redistribution or deployment.

Validation selection: {json.dumps(selection, ensure_ascii=False)}

Complete measured test evaluation: {json.dumps(result, ensure_ascii=False)}

Limitations: eight speakers overlap across source splits; no unseen-speaker
claim is supported. TER is conditional on tone-stripped matching words and must
be read with WER and coverage. Explicit mid tones are not measured. No fitness
for consequential decisions or all Yoruba dialects has been established.
"""
    (best / "README.md").write_text(card)
    token_path = Path.home() / ".cache/huggingface/token"
    token = os.environ.get("HF_TOKEN") or token_path.read_text().strip()
    api = HfApi(token=token)
    if api.whoami()["name"] != a.repo_id.split("/")[0]:
        raise ValueError("Unexpected Hugging Face publishing identity")
    api.create_repo(a.repo_id, exist_ok=True)
    api.upload_folder(
        repo_id=a.repo_id,
        folder_path=best,
        commit_message="Publish selected Yoruba ASR checkpoint with measured evaluation",
    )


if __name__ == "__main__":
    main()
