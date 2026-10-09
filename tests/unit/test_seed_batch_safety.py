"""Batch integration regressions; native CUDA subprocesses are simulated here."""

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import pytest
import test_seed_batch as seed_support


@pytest.fixture
def native_case(tmp_path, monkeypatch):
    return seed_support.native_case.__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize("mode", [0o777, 0o755, 0o770])
def test_batch_rejects_unsafe_existing_data_root(tmp_path, mode):
    from gpu_agent.benchmark.batch import _data_lock

    root = tmp_path / "data"
    root.mkdir(mode=mode)
    root.chmod(mode)
    with pytest.raises(ValueError):
        with _data_lock(root):
            pytest.fail("unsafe root accepted")
    assert root.stat().st_mode & 0o777 == mode


@pytest.mark.parametrize("mode", [0o666, 0o644, 0o400])
def test_batch_rejects_unsafe_existing_lock(tmp_path, mode):
    from gpu_agent.benchmark.batch import _data_lock

    root = tmp_path / "data"
    root.mkdir(mode=0o700)
    lock = root / ".seed-batch.lock"
    lock.touch(mode=mode)
    lock.chmod(mode)
    with pytest.raises(ValueError):
        with _data_lock(root):
            pytest.fail("unsafe lock accepted")


def test_batch_rejects_hardlinked_lock(tmp_path):
    from gpu_agent.benchmark.batch import _data_lock

    root = tmp_path / "data"
    root.mkdir(mode=0o700)
    lock = root / ".seed-batch.lock"
    lock.touch(mode=0o600)
    os.link(lock, root / "alias")
    with pytest.raises(ValueError):
        with _data_lock(root):
            pytest.fail("hardlinked lock accepted")


@pytest.mark.parametrize("replacement", ["lock", "directory"])
def test_batch_lease_detects_identity_replacement(tmp_path, replacement):
    from gpu_agent.benchmark.batch import _data_lock

    root = tmp_path / "data"
    root.mkdir(mode=0o700)
    with pytest.raises(ValueError):
        with _data_lock(root):
            if replacement == "lock":
                (root / ".seed-batch.lock").rename(root / "old-lock")
                (root / ".seed-batch.lock").touch(mode=0o600)
            else:
                root.rename(tmp_path / "old-data")
                root.mkdir(mode=0o700)


def test_batch_lock_does_not_create_missing_data_root(tmp_path):
    from gpu_agent.benchmark.batch import _data_lock

    with pytest.raises(ValueError):
        with _data_lock(tmp_path / "missing"):
            pytest.fail("missing operator configuration accepted")
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize(
    "kind,status",
    [
        ("diagnosis", "RUNNING"),
        ("seed_batch", "QUEUED"),
        ("seed_batch", "COMPLETED"),
        ("seed_batch", "FAILED"),
    ],
)
def test_parent_must_be_running_seed_batch(native_case, kind, status):
    store, controller, _, _ = native_case
    seed = seed_support._prepared(native_case)
    parent = store.create_run(kind, binding=controller.binding)
    if status != "QUEUED":
        store.transition(parent.id, "RUNNING", "PREPARING")
        if status != "RUNNING":
            store.transition(parent.id, status, None)
    with pytest.raises(ValueError):
        controller.execute(seed.clean_plan, seed.input_bytes, parent_run_id=parent.id)
    assert not store.children(parent.id)


def test_batch_export_ignores_unlisted_direct_child(native_case, tmp_path):
    from gpu_agent.benchmark.batch_report import export_batch

    store, controller, _, _ = native_case
    summary = seed_support._runner(native_case).run((seed_support._prepared(native_case),))
    extra = store.create_run(
        "case_execution", parent_run_id=summary.batch_run_id, binding=controller.binding
    )
    store.put(extra.id, "unrelated.txt", b"UNLISTED-CHILD-CANARY", "public")
    output = export_batch(store, summary.batch_run_id, tmp_path / "export.zip")
    with zipfile.ZipFile(output) as z:
        assert all(extra.id not in name for name in z.namelist())
        assert b"UNLISTED-CHILD-CANARY" not in b"".join(z.read(n) for n in z.namelist())


def test_batch_export_checks_summary_role_identity(native_case, tmp_path):
    from gpu_agent.benchmark.batch_report import export_batch

    store, _, _, _ = native_case
    summary = seed_support._runner(native_case).run((seed_support._prepared(native_case),))
    batch = store.load(summary.batch_run_id)
    ref = next(r for r in batch.artifact_refs if r.name == "batch/summary.json")
    # Rebind a hash-valid immutable artifact through a forged controller manifest:
    # valid hash alone must not bless cross-role run IDs.
    summary.cases[0].mutant.run_id = summary.cases[0].clean.run_id
    _replace_artifact(store, batch, ref, summary.model_dump_json().encode())
    with pytest.raises(ValueError):
        export_batch(store, batch.id, tmp_path / "forged.zip")
    assert not (tmp_path / "forged.zip").exists()


def test_batch_export_reads_only_from_pinned_run_leases(native_case, tmp_path, monkeypatch):
    from gpu_agent.benchmark.batch_report import export_batch

    store, _, _, _ = native_case
    summary = seed_support._runner(native_case).run((seed_support._prepared(native_case),))

    def path_read_forbidden(*args, **kwargs):
        raise AssertionError("export fell back to a pathname-based RunStore read")

    monkeypatch.setattr(store, "load", path_read_forbidden)
    monkeypatch.setattr(store, "read", path_read_forbidden)
    output = export_batch(store, summary.batch_run_id, tmp_path / "pinned.zip")
    with zipfile.ZipFile(output) as archive:
        assert "summary.json" in archive.namelist()


def test_batch_public_boundary_rejects_evaluator_store(native_case, tmp_path):
    from gpu_agent.benchmark.batch_report import export_batch
    from gpu_agent.store import RunStore

    store, _, _, _ = native_case
    summary = seed_support._runner(native_case).run((seed_support._prepared(native_case),))
    with pytest.raises(ValueError):
        export_batch(
            RunStore(store.root, visibility="evaluator"),
            summary.batch_run_id,
            tmp_path / "private.zip",
        )


def test_role_summary_rederives_oracle_instead_of_trusting_cached_bool(native_case):
    from gpu_agent.benchmark.batch_report import summarize_role

    store, _, execute, _ = native_case
    run_id = execute("clean")
    run = store.load(run_id)
    ref = next(r for r in run.artifact_refs if r.name == "validation/oracle-result.json")
    value = json.loads(store.read(ref))
    value["result"]["passed"] = False
    _replace_artifact(store, run, ref, json.dumps(value).encode())
    result = summarize_role(store, run_id, seed_support._prepared(native_case).clean_plan)
    assert result.oracle_passed is True
    assert result.reason_code == "ORACLE_BINDING_MISMATCH"
    # Raw input/output, not the cached convenience bool, determine the display.


def test_seed_batch_source_does_not_provision_or_override_family():
    import ast

    import gpu_agent.benchmark.batch as batch

    tree = ast.parse(Path(batch.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr == "provision" for n in calls)
    assert "_family_environment" not in Path(batch.__file__).read_text()


def _replace_artifact(store, run, ref, content):
    path = store.root / ref.relative_path
    path.chmod(0o600)
    path.write_bytes(content)
    replacement = ref.model_copy(
        update={"sha256": hashlib.sha256(content).hexdigest(), "byte_count": len(content)}
    )
    store._save(
        run.model_copy(
            update={
                "artifact_refs": [replacement if r.id == ref.id else r for r in run.artifact_refs]
            }
        )
    )


def test_parent_cannot_finish_between_child_creation_and_gpu_start(native_case):
    store, controller, _, _ = native_case
    seed = seed_support._prepared(native_case)
    parent = store.create_run("seed_batch", binding=controller.binding)
    store.transition(parent.id, "RUNNING", "VERIFYING")

    def finish_parent(run_id):
        store.transition(parent.id, "COMPLETED", None)

    with pytest.raises(ValueError):
        controller.execute(
            seed.clean_plan, seed.input_bytes, parent_run_id=parent.id, on_created=finish_parent
        )
    children = store.children(parent.id)
    assert len(children) == 1
    assert not any(r.name.startswith("build/") for r in children[0].artifact_refs)


def test_parent_must_remain_running_through_child_finalization(native_case):
    store, controller, _, _ = native_case
    seed = seed_support._prepared(native_case)
    parent = store.create_run("seed_batch", binding=controller.binding)
    store.transition(parent.id, "RUNNING", "VERIFYING")
    original_cleanup = controller.backend.cleanup

    def finish_parent_before_child(handle):
        result = original_cleanup(handle)
        store.transition(parent.id, "COMPLETED", None)
        return result

    controller.backend.cleanup = finish_parent_before_child
    with pytest.raises(ValueError, match="parent must be a RUNNING seed_batch"):
        controller.execute(seed.clean_plan, seed.input_bytes, parent_run_id=parent.id)
    child = store.children(parent.id)[0]
    assert child.status == "RUNNING"
    assert child.current_phase == "EXECUTING"


def test_boundary_failure_does_not_write_through_path_based_store(native_case, monkeypatch):
    store, controller, _, _ = native_case
    seed = seed_support._prepared(native_case)
    parent = store.create_run("seed_batch", binding=controller.binding)
    store.transition(parent.id, "RUNNING", "VERIFYING")
    original_load = store.load
    child_id = None
    boundary_checks = 0

    def created(actual_id):
        nonlocal child_id
        child_id = actual_id

        def forbidden(*args, **kwargs):
            raise AssertionError("boundary failure attempted a pathname-based store write")

        monkeypatch.setattr(store, "load", forbidden)
        monkeypatch.setattr(store, "transition", forbidden)

    def boundary_failure():
        nonlocal boundary_checks
        boundary_checks += 1
        if boundary_checks > 1:
            raise RuntimeError("directory identity changed")

    with pytest.raises(RuntimeError, match="directory identity changed"):
        controller.execute(
            seed.clean_plan,
            seed.input_bytes,
            parent_run_id=parent.id,
            on_created=created,
            integrity_check=boundary_failure,
        )
    assert child_id is not None
    assert original_load(child_id).status == "RUNNING"


@pytest.mark.parametrize("store_name", ["runs", "corpus"])
def test_batch_uses_existing_family_without_provisioning(tmp_path, monkeypatch, store_name):
    from gpu_agent.benchmark import batch
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.execution.isolated import Availability

    repo = tmp_path / "repo"
    repo.mkdir()
    for directory in ("benchmarks", "containers"):
        shutil.copytree(seed_support.ROOT / directory, repo / directory)
    for args in (
        ["init", "-q"],
        ["config", "user.name", "Test"],
        ["config", "user.email", "test@localhost"],
        ["add", "."],
        ["commit", "-qm", "fixture"],
    ):
        import subprocess

        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    family = CorpusFamily.provision(
        tmp_path / "existing-authority",
        public_store=data / store_name,
        evaluator_store=data / "evaluator",
        repository=repo,
    )
    namespace = family.namespace_hash
    monkeypatch.setenv("GPU_AGENT_CORPUS_FAMILY_ROOT", str(family.root))
    monkeypatch.setattr(batch, "PACKAGE_REPOSITORY", repo)
    monkeypatch.setattr(
        batch.IsolatedGPUBackend,
        "availability",
        lambda self: Availability(ready=False, reason="OFFLINE_TEST"),
    )
    prepared = batch.prepare_seed_batch(repo, data, ("case_0001",))
    first = batch.run_public_seeds(prepared)
    assert first.stopped_reason == "BACKEND_UNAVAILABLE"
    assert not first.all_passed
    assert os.environ["GPU_AGENT_CORPUS_FAMILY_ROOT"] == str(family.root)
    assert CorpusFamily.open(family.root).namespace_hash == namespace
    assert family.ledger.committed_through() == []
    assert not (data / "controller").exists()


def test_instrumented_summary_rederives_findings_from_raw_log(native_case):
    from gpu_agent.benchmark.batch_report import summarize_role

    store, _, execute, _ = native_case
    run_id = execute("mutant")
    run = store.load(run_id)
    ref = next(r for r in run.artifact_refs if r.name == "validation/sanitizer-00.json")
    value = json.loads(store.read(ref))
    value["findings"] = []
    value["check_outcome"] = "CLEAN"
    _replace_artifact(store, run, ref, json.dumps(value).encode())
    result = summarize_role(store, run_id, seed_support._prepared(native_case).mutant_plan)
    assert result.target_detections == [True]
    assert result.sanitizer_outcomes == ["FINDING"]


def test_role_summary_rejects_same_run_but_wrong_sanitizer_log(native_case):
    from gpu_agent.benchmark.batch_report import summarize_role

    store, _, execute, _ = native_case
    run_id = execute("mutant")
    run = store.load(run_id)
    ref = next(r for r in run.artifact_refs if r.name == "validation/sanitizer-00.json")
    value = json.loads(store.read(ref))
    value["tool_result"]["stderr_artifact"] = value["tool_result"]["stdout_artifact"]
    _replace_artifact(store, run, ref, json.dumps(value).encode())
    result = summarize_role(store, run_id, seed_support._prepared(native_case).mutant_plan)
    assert result.failure_stage == "EVIDENCE"
    assert result.reason_code == "SANITIZER_BINDING_MISMATCH"
