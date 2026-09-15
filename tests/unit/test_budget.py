"""Physical reservations reject work before a budgeted operation starts."""

from concurrent.futures import ThreadPoolExecutor

import pytest


def test_final_calls_are_reserved():
    from gpu_agent.agent.policy import BudgetExceeded, BudgetLedger

    ledger = BudgetLedger()
    for _ in range(4):
        ledger.reserve("planner_llm")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("planner_llm")
    ledger.reserve("diagnosis_llm")
    ledger.reserve("patch_llm")


def test_concurrent_reservations_cannot_oversubscribe():
    from gpu_agent.agent.policy import BudgetExceeded, BudgetLedger

    ledger = BudgetLedger()

    def reserve() -> bool:
        try:
            ledger.reserve("run_memcheck")
            return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=12) as pool:
        accepted = list(pool.map(lambda _: reserve(), range(20)))
    assert accepted.count(True) == 4
    assert accepted.count(False) == 16


def test_retry_and_wall_time_are_bounded_and_audited():
    from gpu_agent.agent.policy import BudgetExceeded, BudgetLedger

    times = iter([0.0, 0.0, 0.0, 0.0, 601.0])
    ledger = BudgetLedger(clock=lambda: next(times))
    first = ledger.reserve("planner_llm")
    ledger.settle(first)
    retry = ledger.reserve("planner_llm", attempt=1)
    ledger.settle(retry, "FAILED")
    with pytest.raises(BudgetExceeded, match="RETRY_EXHAUSTED"):
        ledger.reserve("planner_llm", attempt=1)
    with pytest.raises(BudgetExceeded, match="WALL_TIME_EXHAUSTED"):
        ledger.reserve("run_memcheck")
    states = [event["state"] for event in ledger.audit]
    assert {"ATTEMPTED", "REJECTED", "STARTED", "COMPLETED", "FAILED"} <= set(states)


def test_settlement_is_single_use():
    from gpu_agent.agent.policy import BudgetLedger

    ledger = BudgetLedger()
    reservation = ledger.reserve("inspect_source")
    ledger.settle(reservation)
    with pytest.raises(ValueError, match="already settled"):
        ledger.settle(reservation)
