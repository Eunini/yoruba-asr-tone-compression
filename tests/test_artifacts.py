from eval.artifacts import fingerprint


def test_model_fingerprints_survive_transport_and_detect_changes(tmp_path):
    left, right = tmp_path / "kaggle", tmp_path / "downloaded"
    left.mkdir()
    right.mkdir()
    for folder in [left, right]:
        (folder / "model.safetensors").write_bytes(b"test-weights")
        (folder / "config.json").write_text('{"language":"yo"}')
    assert fingerprint(left) == fingerprint(right)
    (right / "model.safetensors").write_bytes(b"different-weights")
    assert fingerprint(left) != fingerprint(right)
