import numpy as np
import pytest

from eval.audio import chunks


@pytest.mark.parametrize("length", [1, 399, 16000, 480000, 480001, 959999, 960001])
def test_boundary_chunks_cover_audio_without_tiny_final_fragments(length):
    source = np.arange(length)
    parts = list(chunks(source))
    np.testing.assert_array_equal(np.concatenate(parts), source)
    assert all(0 < len(p) <= 480000 for p in parts)
    if len(parts) > 1:
        assert len(parts[-1]) >= 16000
