import pytest

from eval.paired import compare


def row(text, item_id="one", reference="ọkọ̀ ilé"):
    return {"id": item_id, "reference": reference, "hypothesis": text, "duration_seconds": 2.0}


def test_losing_a_word_does_not_artificially_improve_paired_tone_accuracy():
    result = compare([row("ọkọ̀ ile")], [row("ọkọ̀ nkan")], replicates=20)
    assert result["control_marginal"]["tone_error_rate"] == 0.5
    assert result["quantized_marginal"]["tone_error_rate"] == 0
    assert result["common_word_coverage"] == 0.5
    assert result["common_tone_coverage"] == 0.5
    assert result["common_ter_delta"] == 0
    assert result["paired_utterance_bootstrap_95ci"] == [0, 0]


def test_order_is_irrelevant_but_ids_and_references_must_match():
    a = [row("ọkọ̀ ilé"), row("ilé", "two", "ilé")]
    b = [row("ile", "two", "ilé"), row("ọkọ ilé")]
    result = compare(a, b, replicates=20)
    assert result["common_ter_delta"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        compare(a, b[:1])
    with pytest.raises(ValueError):
        compare(a, [row("ọkọ", reference="different"), b[0]])


def test_no_common_tones_is_undefined_not_perfect_accuracy():
    result = compare([row("different")], [row("other")], replicates=20)
    assert result["common_ter_delta"] is None
    assert result["paired_utterance_bootstrap_95ci"] is None
    assert result["bootstrap_valid_replicates"] == 0
