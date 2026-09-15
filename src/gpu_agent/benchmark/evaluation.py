"""Serial, cost-capped evaluation records and blinded reviewer projections."""

import random
from collections.abc import Callable
from typing import Literal

from pydantic import Field

from gpu_agent.benchmark.metrics import Score
from gpu_agent.execution.models import ExecutionModel

EvaluationMode = Literal["A", "B", "C", "D", "E"]


class EvaluationRecord(ExecutionModel):
    record_id: str
    case_id: str
    template_id: str
    mode: EvaluationMode
    repeat: int = Field(ge=0)
    input_hash: str
    evidence_hash: str
    executed_checks: dict[str, str]
    status: Literal["COMPLETED", "FAILED", "TIMEOUT", "INCONCLUSIVE"]
    diagnosis: dict[str, object]
    patch_hash: str | None = None
    oracle_passed: bool | None = None
    verdict: str | None = None
    regression_detected: bool = False
    usage: dict[str, int | None] = Field(default_factory=dict)
    latency_ms: float = Field(ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    failure_reason: str | None = None
    should_be_inconclusive: bool | None = None
    score: Score | None = None

    def blind(self) -> dict[str, object]:
        return {
            "blind_id": self.record_id,
            "diagnosis": self.diagnosis,
            "evidence_hash": self.evidence_hash,
        }


class EvaluationManifest(ExecutionModel):
    schema_version: Literal[1] = 1
    modes: list[EvaluationMode]
    split: Literal["development", "holdout"]
    repeats: int = Field(ge=3)
    random_seed: int
    records: list[EvaluationRecord]
    stopped_reason: str | None = None


class EvaluationRunner:
    def __init__(
        self,
        case_ids: dict[str, str],
        execute: Callable[[str, str, EvaluationMode, int], EvaluationRecord],
        *,
        max_cost_usd: float | None,
        random_seed: int = 20260915,
    ) -> None:
        self.case_ids, self.execute = dict(case_ids), execute
        self.max_cost_usd, self.random_seed = max_cost_usd, random_seed

    def run(
        self,
        mode: EvaluationMode | Literal["all"],
        split: Literal["development", "holdout"],
        repeats: int,
    ) -> EvaluationManifest:
        if repeats < 3:
            raise ValueError("evaluation requires at least three repeats")
        modes: list[EvaluationMode] = ["A", "B", "C", "D", "E"] if mode == "all" else [mode]
        schedule = [
            (case, template, item, repeat)
            for repeat in range(repeats)
            for case, template in self.case_ids.items()
            for item in modes
        ]
        random.Random(self.random_seed).shuffle(schedule)
        records: list[EvaluationRecord] = []
        spent = 0.0
        stopped = None
        for case, template, item, repeat in schedule:
            if self.max_cost_usd is None:
                stopped = "COST_CAP_REQUIRED"
                break
            record = self.execute(case, template, item, repeat)
            records.append(record)
            if record.cost_usd is None:
                stopped = "COST_UNKNOWN"
                break
            spent += record.cost_usd
            if spent > self.max_cost_usd:
                stopped = "COST_CAP_EXCEEDED"
                break
        return EvaluationManifest(
            modes=modes,
            split=split,
            repeats=repeats,
            random_seed=self.random_seed,
            records=records,
            stopped_reason=stopped,
        )
