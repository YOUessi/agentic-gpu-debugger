"""Record-only estimates from explicit rates; never authorizes or limits calls."""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from gpu_agent.agent.provider import Invocation


class AccountingRates(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    source: str = Field(min_length=1)
    input_usd_per_million: Decimal = Field(ge=0, allow_inf_nan=False)
    output_usd_per_million: Decimal = Field(ge=0, allow_inf_nan=False)
    cached_input_usd_per_million: Decimal | None = Field(default=None, ge=0, allow_inf_nan=False)


def summarize_calls(calls: list[Invocation], rates: AccountingRates | None) -> dict[str, object]:
    """Count each physical invocation once. Partial estimates are not total costs."""
    if len({call.invocation_id for call in calls}) != len(calls):
        raise ValueError("duplicate physical invocation")
    known = Decimal(0)
    unknown = 0
    inputs = outputs = 0
    for call in calls:
        usage = call.usage
        if usage is not None:
            inputs += usage.input_tokens or 0
            outputs += usage.output_tokens or 0
        if (
            rates is None
            or call.state == "STARTED"
            or call.configured_model != rates.model
            or (call.response_model is not None and call.response_model != rates.model)
            or usage is None
            or usage.input_tokens is None
            or usage.output_tokens is None
        ):
            unknown += 1
            continue
        cached = usage.cached_tokens
        if rates.cached_input_usd_per_million is not None and cached is None:
            unknown += 1
            continue
        if (
            usage.input_tokens < 0
            or usage.output_tokens < 0
            or (cached is not None and not 0 <= cached <= usage.input_tokens)
        ):
            raise ValueError("invalid token usage")
        if cached and rates.cached_input_usd_per_million is None:
            # Do not silently charge cached input at an unverified rate.
            unknown += 1
            continue
        known += (
            Decimal(usage.input_tokens - (cached or 0)) * rates.input_usd_per_million
            + Decimal(cached or 0) * (rates.cached_input_usd_per_million or Decimal(0))
            + Decimal(usage.output_tokens) * rates.output_usd_per_million
        ) / Decimal(1_000_000)
    return {
        "physical_calls": len(calls),
        "known_input_tokens": inputs,
        "known_output_tokens": outputs,
        "unknown_cost_calls": unknown,
        "known_estimated_cost_usd": str(known),
        "estimated_cost_usd": str(known) if unknown == 0 else None,
        "rates": rates.model_dump(mode="json") if rates else None,
        "cost_kind": "estimate_not_provider_invoice",
    }
