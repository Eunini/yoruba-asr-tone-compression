from enum import Enum
from types import SimpleNamespace

import pytest

from train.collect_gpu import kernel_state


class Status(Enum):
    RUNNING = 1


@pytest.mark.parametrize(
    "response,expected",
    [
        (SimpleNamespace(status=Status.RUNNING), "RUNNING"),
        ('{"status":"RUNNING","failureMessage":null}', "RUNNING"),
        ({"status": "complete"}, "COMPLETE"),
    ],
)
def test_collection_accepts_sdk_enum_objects_and_serialized_responses(response, expected):
    assert kernel_state(response) == expected
