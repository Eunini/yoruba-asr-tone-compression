import unicodedata

import pytest

from eval.metrics import score, strip_tones


@pytest.mark.parametrize(
    "ref,hyp,wer,ter",
    [
        ("Mo fẹ́ ẹja", "Mo fẹ́ ẹja", 0, 0),
        ("Mo fẹ́ ẹja", "Mo fẹ ẹja", 0, 1),
        ("ó lọ", "ò lọ", 0, 1),
        ("ó lọ", "ó wá", 0.5, 0),
        ("ọmọ ṣé", "omo sé", 1, None),
    ],
)
def test_five_handwritten_examples(ref, hyp, wer, ter):
    result = score([ref], [hyp])
    assert result["wer"] == wer
    assert result["tone_error_rate"] == ter


def test_underdots_and_unicode_equivalence():
    assert strip_tones("ẹ́ ọ̀ ṣ") == "ẹ ọ ṣ"
    assert strip_tones(unicodedata.normalize("NFD", "ẹ́ ọ̀ ṣ")) == "ẹ ọ ṣ"
    assert score(["ẹ́"], [unicodedata.normalize("NFD", "ẹ́")])["tone_error_rate"] == 0


def test_deletions_cannot_look_perfect():
    result = score(["ó lọ"], [""])
    assert result["wer"] == 1
    assert result["tone_error_rate"] is None
    assert result["matched_tone_coverage"] == 0


def test_added_tones_and_corpus_weighting():
    result = score(["ó", "mo"], ["o", "mó"])
    assert result["tone_edits"] == 2
    assert result["tone_error_rate"] == 2  # Insertions allow rates above 100%.
    assert score(["mo"], ["mo"])["tone_error_rate"] is None
    with pytest.raises(ValueError):
        score(["a"], [])
