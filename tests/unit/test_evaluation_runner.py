import pytest
from pydantic import ValidationError


def test_public_evaluation_record_requires_native_lineage():
    from gpu_agent.benchmark.evaluation import PublicEvaluationRecord

    with pytest.raises(ValidationError, match="lineage"):
        PublicEvaluationRecord(
            record_id="f" * 32,
            case_id="case_0001",
            template_id="index",
            mode="A",
            repeat=0,
            input_hash="a" * 64,
            evidence_hash="b" * 64,
            executed_checks={},
            status="INCONCLUSIVE",
            diagnosis={"diagnostic_outcome": "INCONCLUSIVE", "limitations": ["NO_FINDING"]},
            latency_ms=1,
            cost_usd=0,
        )
