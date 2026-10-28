import json

import pytest

from eval.artifacts import fingerprint
from train.publish import TEST_SET, prepare_release


def release_inputs(tmp_path):
    root = tmp_path / "checkpoint"
    best = root / "best"
    best.mkdir(parents=True)
    (best / "model.safetensors").write_bytes(b"selected-weights-fixture")
    (best / "config.json").write_text("{}")
    (best / "training_args.bin").write_bytes(b"pickle-state-must-not-be-published")
    selection = {
        "checkpoint": "checkpoint-800",
        "base_model": "openai/whisper-tiny",
        "base_model_revision": "pinned",
        "artifacts": fingerprint(best),
    }
    (root / "selection.json").write_text(json.dumps(selection))
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "status": "measured",
                "partial": False,
                "n": 177,
                "evaluation_set_sha256": TEST_SET,
                "artifacts": selection["artifacts"],
            }
        )
    )
    card = tmp_path / "card.md"
    card.write_text(
        "---\nlicense: cc-by-sa-4.0\nbase_model: openai/whisper-tiny\n---\nResearch only.\n"
    )
    notices = tmp_path / "notices"
    notices.mkdir()
    for name in [
        "LICENSE",
        "LICENSE-BASE-APACHE-2.0.txt",
        "LICENSE-UPSTREAM-WHISPER-MIT.txt",
        "NOTICE.md",
    ]:
        (notices / name).write_text("test notice fixture")
    return root, evaluation, card, notices


def test_staging_preserves_selected_checkpoint_and_excludes_pickle(tmp_path):
    args = release_inputs(tmp_path)
    before = fingerprint(args[0] / "best")
    target = tmp_path / "release"
    result = prepare_release(*args, target, "cc-by-sa-4.0")
    assert fingerprint(args[0] / "best") == before
    assert not (args[0] / "best/README.md").exists()
    assert not (target / "training_args.bin").exists()
    assert result["artifacts_excluded"] == ["training_args.bin"]
    assert (target / "model.safetensors").read_bytes() == b"selected-weights-fixture"
    assert (target / "NOTICE.md").exists()


@pytest.mark.parametrize("change", ["test", "weights", "card"])
def test_wrong_test_changed_weights_or_wrong_card_prevents_release(tmp_path, change):
    args = release_inputs(tmp_path)
    if change == "test":
        result = json.loads(args[1].read_text())
        result["evaluation_set_sha256"] = "other-test"
        args[1].write_text(json.dumps(result))
    elif change == "weights":
        (args[0] / "best/model.safetensors").write_bytes(b"changed")
    else:
        args[2].write_text(args[2].read_text().replace("cc-by-sa-4.0", "mit"))
    target = tmp_path / "release"
    with pytest.raises(ValueError):
        prepare_release(*args, target, "cc-by-sa-4.0")
    assert not target.exists()
