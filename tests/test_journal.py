import json

import pytest

from eval.journal import prepare


ROWS = [
    {"id": "one", "reference": "ẹ káàrọ̀", "duration_seconds": 2.0},
    {"id": "two", "reference": "ọkọ̀", "duration_seconds": 1.0},
]


def record():
    return {**ROWS[0], "hypothesis": "ẹ kaaro", "processing_seconds": 0.2}


def test_resume_preserves_saved_prefix_and_rejects_changed_identity(tmp_path):
    identity = {"weights": "original", "threads": 2}
    assert prepare(tmp_path, identity, ROWS, False) == []
    path = tmp_path / "predictions.jsonl"
    original = (json.dumps(record()) + "\n").encode()
    path.write_bytes(original)
    assert prepare(tmp_path, identity, ROWS, True) == [record()]
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="identity changed"):
        prepare(tmp_path, {**identity, "weights": "changed"}, ROWS, True)
    with pytest.raises(FileExistsError):
        prepare(tmp_path, identity, ROWS, False)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "bad",
    [
        b'{"id":',
        b"{}\n",
        (json.dumps({**record(), "id": "two"}) + "\n").encode(),
        (json.dumps({**record(), "processing_seconds": float("nan")}) + "\n").encode(),
    ],
)
def test_resume_never_silently_repairs_corrupt_or_mismatched_answers(tmp_path, bad):
    prepare(tmp_path, {}, ROWS, False)
    path = tmp_path / "predictions.jsonl"
    path.write_bytes(bad)
    with pytest.raises(ValueError):
        prepare(tmp_path, {}, ROWS, True)
    assert path.read_bytes() == bad


def test_legacy_results_cannot_gain_invented_provenance(tmp_path):
    (tmp_path / "predictions.jsonl").write_text(json.dumps(record()) + "\n")
    with pytest.raises(ValueError, match="legacy"):
        prepare(tmp_path, {}, ROWS, True)
