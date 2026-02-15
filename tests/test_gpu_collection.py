import pytest

from train.collect_gpu import kernel_state


@pytest.mark.parametrize(
    "response,expected",
    [('{"status":"RUNNING","failureMessage":null}', "RUNNING"), ({"status": "complete"}, "COMPLETE")],
)
def test_collection_accepts_real_sdk_json_text_and_object_responses(response, expected):
    assert kernel_state(response) == expected
