from datetime import datetime, timedelta, timezone

import pytest

from serve.cloud_benchmark import preflight_price, state_is_empty


def price(age_hours=0, hourly=0.10404767123287671):
    return {
        "checked_at": (datetime.now(timezone.utc) - timedelta(hours=age_hours)).isoformat(),
        "region": "us-east-1",
        "topologies": {"1xm7i-flex.large": {"cluster_usd_per_hour": hourly}},
    }


def test_cloud_budget_uses_paid_list_price_even_when_credits_exist():
    assert preflight_price(price(), 24, 6) == pytest.approx(0.10404767123287671)
    with pytest.raises(ValueError, match="budget"):
        preflight_price(price(), 48, 2)


@pytest.mark.parametrize("hours", [0, -1, 49])
def test_unbounded_or_invalid_cloud_lifetime_is_rejected(hours):
    with pytest.raises(ValueError):
        preflight_price(price(), hours, 100)


def test_stale_or_wrong_region_price_cannot_start_compute():
    with pytest.raises(ValueError, match="Refresh"):
        preflight_price(price(age_hours=49), 24, 6)
    wrong = price()
    wrong["region"] = "eu-west-1"
    with pytest.raises(ValueError, match="Refresh"):
        preflight_price(wrong, 24, 6)


def test_fresh_backend_is_empty_but_existing_resources_or_corruption_block_reuse(tmp_path):
    state = tmp_path / "terraform.tfstate"
    assert state_is_empty(state)
    state.write_text('{"resources": []}')
    assert state_is_empty(state)
    state.write_text('{"resources": [{"type": "aws_instance"}]}')
    assert not state_is_empty(state)
    state.write_text('incomplete')
    with pytest.raises(ValueError):
        state_is_empty(state)
