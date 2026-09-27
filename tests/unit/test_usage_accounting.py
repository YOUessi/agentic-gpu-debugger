from decimal import Decimal

import pytest

from gpu_agent.agent.provider import Invocation, Usage
from gpu_agent.contracts import now
from gpu_agent.usage_accounting import AccountingRates, summarize_calls


def call(usage=None, **updates):
    return Invocation(
        invocation_id="a" * 32,
        run_id="b" * 32,
        kind="plan",
        attempt=0,
        state="COMPLETED",
        started_at=now(),
        configured_model="test",
        endpoint_host="example.org",
        client_request_id="c" * 32,
        usage=usage,
    ).model_copy(update=updates)


def rates(**updates):
    return AccountingRates(
        model="test",
        source="synthetic test rates",
        input_usd_per_million=Decimal(2),
        output_usd_per_million=Decimal(4),
        **updates,
    )


def test_exact_estimate_and_missing_cost():
    c = call(Usage(input_tokens=1000, output_tokens=500))
    assert summarize_calls([c], rates())["estimated_cost_usd"] == "0.004"
    assert summarize_calls([c], None)["estimated_cost_usd"] is None
    assert summarize_calls([call()], rates())["unknown_cost_calls"] == 1
    assert summarize_calls([], None)["estimated_cost_usd"] == "0"


def test_cache_model_mismatch_and_duplicate():
    c = call(Usage(input_tokens=1000, output_tokens=500, cached_tokens=500))
    assert summarize_calls([c], rates())["estimated_cost_usd"] is None
    assert (
        summarize_calls([c], rates(cached_input_usd_per_million=Decimal(1)))["estimated_cost_usd"]
        == "0.0035"
    )
    assert (
        summarize_calls([c.model_copy(update={"configured_model": "other"})], rates())[
            "estimated_cost_usd"
        ]
        is None
    )
    with pytest.raises(ValueError, match="duplicate"):
        summarize_calls([c, c], rates())


def test_partial_cost_is_not_reported_as_total():
    report = summarize_calls(
        [
            call(Usage(input_tokens=1000, output_tokens=500)),
            call(invocation_id="d" * 32, state="UNCERTAIN"),
        ],
        rates(),
    )
    assert report["known_estimated_cost_usd"] == "0.004"
    assert report["estimated_cost_usd"] is None
