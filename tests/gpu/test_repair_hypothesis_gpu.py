"""Reproduce case_0022 candidate-one evidence gap with real CUDA/Sanitizer.

Only the prior diagnosis and candidate are scripted. Build, functional oracle and
both Sanitizers execute on Tang's actual GPU. No model API or private oracle.
"""

import hashlib
import json
from pathlib import Path

import pytest

from gpu_agent.agent.models import (
    AgentBudget,
    DiagnosisResult,
    FinishAction,
    PublicEvidence,
    PublicRepairContext,
)
from gpu_agent.agent.policy import decide_action, missing_evidence
from gpu_agent.agent.rule_router import RuleRouter
from gpu_agent.contracts import CurrentPhase
from gpu_agent.execution.isolated import IsolatedGPUBackend
from gpu_agent.execution.models import (
    BuildRequest,
    ExecutionRequest,
    SanitizerRequest,
    SanitizerTool,
    WorkspaceRequest,
)
from gpu_agent.public_task import PublicTask, check_public_output
from gpu_agent.store import RunStore

pytestmark = [pytest.mark.gpu, pytest.mark.container]


def test_case22_numeric_mismatch_must_recheck_original_race_on_real_gpu(tmp_path, request):
    repo = Path(__file__).resolve().parents[2]
    task_root = repo / "benchmarks/public/case_0022/public_input"
    original = (task_root / "kernel.cu").read_text()
    start = "        __syncthreads();\n        tile[lane] = value;"
    assert original.count(start) == 1
    failed = original.replace(start, "        tile[lane] = value;\n        __syncthreads();", 1)
    task = PublicTask.model_validate_json((task_root / "task.json").read_bytes())
    assert task.source_sha256 == hashlib.sha256(original.encode()).hexdigest()

    source_root = tmp_path / "source"
    source_root.mkdir()
    sources = {"kernel.cu": failed.encode()}
    for name in ("vector_api.h", "vector_io.cpp", "json.hpp"):
        path = (
            repo / "benchmarks/harness/vendor/json.hpp"
            if name == "json.hpp"
            else repo / "benchmarks/harness" / name
        )
        sources[name] = path.read_bytes()
    store_root = request.config.getoption("--gpu-run-root")
    store = RunStore(Path(store_root) if store_root else tmp_path / "public-runs")
    backend = IsolatedGPUBackend(store, source_root, tmp_path / "tasks")
    ready = backend.availability()
    if not ready.ready:
        pytest.skip(ready.reason)
    for name, data in sources.items():
        IsolatedGPUBackend._write_snapshot(source_root / name, data)

    run = store.create_run("diagnosis")
    store.transition(run.id, "RUNNING", "PREPARING")
    handle = None
    try:
        handle = backend.prepare(
            WorkspaceRequest(
                run_id=run.id,
                source_manifest={n: hashlib.sha256(b).hexdigest() for n, b in sources.items()},
                trust_level="UNTRUSTED",
            )
        )
        store.transition(run.id, "RUNNING", "COMPILING")
        build = backend.build(BuildRequest(workspace_id=handle.id, timeout_seconds=120))
        assert build.success
        stdin = (task_root / "input.json").read_bytes()
        input_ref = store.put(run.id, "public-input.json", stdin, "public")
        store.transition(run.id, "RUNNING", "EXECUTING")
        execution = backend.run(
            ExecutionRequest(workspace_id=handle.id, stdin_ref=input_ref, timeout_seconds=30)
        )
        assert execution.runtime_status == "SUCCESS"
        output_check = check_public_output(task, stdin, store.read(execution.output_ref))
        assert output_check == "NUMERIC_MISMATCH"
        memcheck = backend.run_sanitizer(
            SanitizerRequest(
                workspace_id=handle.id,
                stdin_ref=input_ref,
                tool="memcheck",
                timeout_seconds=120,
            )
        )
        assert memcheck.completed and memcheck.check_outcome == "CLEAN"

        previous = DiagnosisResult(
            diagnostic_outcome="DIAGNOSED",
            failure_family="shared_memory_race",
            root_cause="Prior public racecheck implicated staged shared-memory access.",
        )
        context = PublicRepairContext(
            repair_round=1,
            original_source_sha256=hashlib.sha256(original.encode()).hexdigest(),
            candidate_source_sha256=hashlib.sha256(failed.encode()).hexdigest(),
            previous_diagnosis_source_sha256=hashlib.sha256(original.encode()).hexdigest(),
            previous_diagnosis=previous,
            public_checks={"functional": "NUMERIC_MISMATCH"},
            public_feedback=[],
            public_functional_failure=True,
        )
        before = PublicEvidence(
            repair_context=context,
            sanitizer_outcomes={SanitizerTool.MEMCHECK: memcheck.check_outcome},
        )
        assert "racecheck_outcome" in missing_evidence(before)
        assert not decide_action(
            FinishAction(), before, AgentBudget(), CurrentPhase.DIAGNOSING, set()
        ).allowed
        assert RuleRouter().next_action(before, AgentBudget()).action_type == "run_racecheck"

        racecheck = backend.run_sanitizer(
            SanitizerRequest(
                workspace_id=handle.id,
                stdin_ref=input_ref,
                tool="racecheck",
                timeout_seconds=120,
            )
        )
        assert racecheck.completed
        assert racecheck.check_outcome in {"FINDING", "CLEAN"}
        after = before.model_copy(
            update={
                "sanitizer_outcomes": {
                    **before.sanitizer_outcomes,
                    SanitizerTool.RACECHECK: racecheck.check_outcome,
                }
            }
        )
        assert "racecheck_outcome" not in missing_evidence(after)
        print(
            json.dumps(
                {
                    "scope": "case22_prior_hazard_native_GPU_gate",
                    "run_id": run.id,
                    "functional": output_check,
                    "memcheck": memcheck.check_outcome,
                    "racecheck": racecheck.check_outcome,
                    "must_acquire_racecheck": True,
                    "model_calls": 0,
                },
                sort_keys=True,
            )
        )
    finally:
        if handle is not None:
            backend.cleanup(handle)
