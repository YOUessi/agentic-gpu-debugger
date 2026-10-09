import hashlib
import json
from pathlib import Path

import pytest

from gpu_agent.agent.policy import LLMCallGate
from gpu_agent.execution.process import ProcessCapture
from gpu_agent.public_task import PublicTask, check_public_output, load_public_task
from gpu_agent.repair import self_check


def task(algorithm="rotate-add-cpu-v1"):
    return PublicTask(source_sha256="0" * 64, algorithm=algorithm)


def output(values):
    return json.dumps({"dtype": "float32", "shape": [len(values)], "values": values}).encode()


@pytest.mark.parametrize(
    "values,expected",
    [([5, 7, 5], "PASSED"), ([4, 6, 8], "NUMERIC_MISMATCH"), ([5, 7, 4], "NUMERIC_MISMATCH")],
)
def test_rotate_spec_rejects_safe_wrong_computations(values, expected):
    stdin = b'{"n":3,"a":[1,2,3],"b":[3,4,4]}'
    assert check_public_output(task(), stdin, output(values)) == expected


def test_group_sum_rejects_shifted_input():
    stdin = json.dumps({"n": 33, "a": list(range(33)), "b": [1] * 33}).encode()
    assert (
        check_public_output(task("warp-reduce-cpu-v1"), stdin, output([528] * 32 + [33]))
        == "PASSED"
    )
    assert (
        check_public_output(task("warp-reduce-cpu-v1"), stdin, output([560] * 32 + [33]))
        == "NUMERIC_MISMATCH"
    )


@pytest.mark.parametrize(
    "operation,status",
    [
        ("runtime", "FAILED"),
        ("timeout", "UNAVAILABLE"),
        ("wrong", "FAILED"),
        ("clean", "PASSED"),
        ("instrumented_wrong", "FAILED"),
    ],
)
def test_native_public_checks(oob_service, operation, status):
    service, _, _ = oob_service
    operations = []

    class Backend(service._backend_factory):
        def _container(self, path, op, timeout, *, stdin=b"", cancel=None):
            operations.append(op)
            if op.startswith("build"):
                return ProcessCapture(0, b"", b"", False), b"binary", b""
            if op == "run" and operation == "runtime":
                return (
                    ProcessCapture(
                        1, b"", b"vector harness: kernel output contains a nonfinite value", False
                    ),
                    b"",
                    b"",
                )
            if op == "run" and operation == "timeout":
                return ProcessCapture(None, b"", b"", True), b"", b""
            values = (
                [99]
                if operation == "wrong" or (op != "run" and operation == "instrumented_wrong")
                else [3]
            )
            log = b"========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n"
            if op == "racecheck":
                log = (
                    b"========= COMPUTE-SANITIZER\n"
                    b"========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"
                )
            return ProcessCapture(0, output(values), b"", False), b"", log

    parent = service.store.create_run("diagnosis")
    checked = self_check(
        service.store,
        parent.id,
        {"kernel.cu": b"int main(){return 0;}\n"},
        b'{"n":1,"a":[1],"b":[2]}',
        Backend,
        LLMCallGate(),
        task(),
    )
    assert checked.status == status
    if operation in {"runtime", "timeout", "wrong"}:
        assert not set(operations) & {"memcheck", "racecheck", "initcheck", "synccheck"}
    if operation == "runtime":
        assert "nonfinite" in checked.feedback[-1]["stderr"]
    if operation == "clean":
        assert checked.checks["functional"] == "PASSED"
        assert checked.checks["synccheck_functional"] == "PASSED"


def test_public_task_binding_and_source_propagation(oob_service, monkeypatch):
    service, provider, source = oob_service
    content = (source / "kernel.cu").read_bytes()
    public_task = PublicTask(
        source_sha256=hashlib.sha256(content).hexdigest(), algorithm="vector-add-cpu-v1"
    )
    (source / "task.json").write_text(public_task.model_dump_json())
    from gpu_agent.agent.orchestrator import public_evidence
    from gpu_agent.repair import PublicCheck

    monkeypatch.setattr(
        "gpu_agent.repair.self_check",
        lambda *a: PublicCheck(run_id="a" * 32, status="UNAVAILABLE", checks={}, feedback=[]),
    )
    run, _ = service.repair(source)
    evidence = public_evidence(service.store, run.id)
    assert evidence.sources[0].functional_requirement == public_task.requirement
    assert "secret-canary" not in evidence.model_dump_json()
    assert any(r.name == "public-task.json" for r in run.artifact_refs)
    with pytest.raises(ValueError, match="binding"):
        load_public_task(source / "kernel.cu", b"changed")


def test_all_public_tasks_bound():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "benchmarks/public").glob("case_*/public_input/kernel.cu"))
    assert len(paths) == 23  # Includes the public case_0000 baseline and two new workloads.
    for kernel in paths:
        assert load_public_task(kernel, kernel.read_bytes()) is not None


def test_public_task_rejects_symlink_and_unregistered_algorithm(tmp_path):
    kernel = tmp_path / "kernel.cu"
    kernel.write_bytes(b"int main(){}")
    contract = tmp_path / "outside.json"
    contract.write_text(task().model_dump_json())
    kernel.with_name("task.json").symlink_to(contract)
    with pytest.raises(ValueError):
        load_public_task(kernel, kernel.read_bytes())
    with pytest.raises(ValueError):
        PublicTask(source_sha256="0" * 64, algorithm="/tmp/model_generated_checker.py")


def test_invalid_public_input_does_not_get_numeric_pass():
    with pytest.raises(ValueError):
        check_public_output(task(), b'{"n":2,"a":[1],"b":[2]}', output([3]))
    assert (
        check_public_output(task(), b'{"n":1,"a":[1],"b":[2]}', b"not-json")
        == "INVALID_NUMERIC_OUTPUT"
    )
