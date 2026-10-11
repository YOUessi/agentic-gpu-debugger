"""Controller-only private batch tests; simulated GPU output is not release evidence."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import test_corpus_registration as corpus_support

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def native_case(tmp_path, monkeypatch):
    return corpus_support.native_case.__wrapped__(tmp_path, monkeypatch)


def _private_fixture(native_case, tmp_path: Path) -> tuple[Path, Path, Path, object]:
    from gpu_agent.benchmark.models import AuthoritativeCaseRegistry

    public_store, public_controller, _, input_bytes = native_case
    source = public_controller.backend.repo_root
    private_root = tmp_path / "controller-private-input"
    workspace_root = tmp_path / "controller-private-workspaces"
    private_root.mkdir(mode=0o700)
    workspace_root.mkdir(mode=0o700)
    private_root.chmod(0o700)
    workspace_root.chmod(0o700)
    for directory in ("clean", "mutant", "harness"):
        target = private_root / directory
        target.mkdir(mode=0o700)
        target.chmod(0o700)
    for relative in (
        "clean/kernel.cu",
        "mutant/kernel.cu",
        "harness/vector_io.cpp",
        "harness/vector_api.h",
        "harness/json.hpp",
    ):
        target = private_root / relative
        target.write_bytes((source / relative).read_bytes())
        target.chmod(0o600)
    input_path = private_root / "input.json"
    input_path.write_bytes(input_bytes)
    input_path.chmod(0o600)

    public_spec = next(iter(public_controller.specs.values()))
    private_spec = public_spec.model_copy(update={"split": "private"})
    registry = private_root / "registry.json"
    registry.write_text(AuthoritativeCaseRegistry(cases=[private_spec]).model_dump_json())
    registry.chmod(0o600)

    def manifest(role: str) -> dict[str, str]:
        names = (
            f"{role}/kernel.cu",
            "harness/vector_io.cpp",
            "harness/vector_api.h",
            "harness/json.hpp",
        )
        return {
            name: hashlib.sha256((private_root / name).read_bytes()).hexdigest() for name in names
        }

    source_manifests = private_root / "source-manifests.json"
    source_manifests.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "case_id": private_spec.case_id,
                        "clean_manifest": manifest("clean"),
                        "mutant_manifest": manifest("mutant"),
                        "input_path": "input.json",
                    }
                ],
            }
        )
    )
    source_manifests.chmod(0o600)

    repository = tmp_path / "repository-checkout"
    repository.mkdir()
    shutil.copytree(ROOT / "containers", repository / "containers")
    for args in (
        ["init", "-q"],
        ["config", "user.name", "Test"],
        ["config", "user.email", "test@localhost"],
        ["add", "."],
        ["commit", "-qm", "private batch fixture"],
    ):
        subprocess.run(["git", *args], cwd=repository, check=True, capture_output=True)
    return repository, private_root, workspace_root, public_store


def _public_tree(store) -> dict[str, str]:
    return {
        str(path.relative_to(store.root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in store.root.rglob("*")
        if path.is_file()
    }


def _use_simulated_backend(monkeypatch, private_batch, native_case, backend_type=None) -> None:
    from gpu_agent.execution.isolated import Availability

    selected = backend_type or type(native_case[1].backend)
    monkeypatch.setattr(
        selected,
        "availability",
        lambda self: Availability(ready=True, reason="OFFLINE_SIMULATION"),
    )
    monkeypatch.setattr(private_batch, "IsolatedGPUBackend", selected)


def test_private_batch_validates_without_registration_and_public_store_is_unchanged(
    native_case, tmp_path, monkeypatch
):
    from gpu_agent.benchmark import private_batch

    repository, private_root, workspace_root, public_store = _private_fixture(native_case, tmp_path)
    before = _public_tree(public_store)
    _use_simulated_backend(monkeypatch, private_batch, native_case)
    prepared = private_batch.prepare_private_batch(repository, private_root, workspace_root)
    projection = private_batch.run_private_batch(prepared, register=False)

    assert projection.status == "COMPLETED"
    assert projection.case_count == 1
    assert set(projection.model_dump()) == {
        "batch_id",
        "case_count",
        "status",
        "private_batch_hash",
    }
    rendered = projection.model_dump_json()
    assert "case_0100" not in rendered
    assert "vector-add-index" not in rendered
    assert "delete-index-guard" not in rendered
    assert _public_tree(public_store) == before
    family = native_case[1].family
    evaluator = family.corpus_store("evaluator")
    parent = evaluator.load(projection.batch_id)
    assert parent.kind == "private_seed_batch"
    assert parent.status == "COMPLETED"
    assert all(ref.visibility == "evaluator" for ref in parent.artifact_refs)
    children = evaluator.children(parent.id)
    assert len(children) == 2
    assert all(ref.visibility == "evaluator" for child in children for ref in child.artifact_refs)
    assert family.ledger.committed_through() == []


def test_private_batch_registers_only_when_explicit(native_case, tmp_path, monkeypatch):
    from gpu_agent.benchmark import private_batch

    repository, private_root, workspace_root, public_store = _private_fixture(native_case, tmp_path)
    before = _public_tree(public_store)
    _use_simulated_backend(monkeypatch, private_batch, native_case)
    projection = private_batch.run_private_batch(
        private_batch.prepare_private_batch(repository, private_root, workspace_root),
        register=True,
    )
    assert projection.status == "COMPLETED"
    transactions = native_case[1].family.ledger.committed_through()
    assert len(transactions) == 1
    assert transactions[0].visibility == "evaluator"
    assert _public_tree(public_store) == before


def test_failed_private_case_retains_evaluator_child_but_projection_stays_opaque(
    native_case, tmp_path, monkeypatch
):
    from gpu_agent.benchmark import private_batch
    from gpu_agent.execution.process import ProcessCapture

    repository, private_root, workspace_root, public_store = _private_fixture(native_case, tmp_path)
    base = type(native_case[1].backend)

    class FailingBackend(base):
        def _container(self, path, operation, timeout, *, stdin=b"", cancel=None):
            if operation == "build":
                return ProcessCapture(1, b"", b"compile failed", False), b"", b""
            return super()._container(path, operation, timeout, stdin=stdin, cancel=cancel)

    before = _public_tree(public_store)
    _use_simulated_backend(monkeypatch, private_batch, native_case, FailingBackend)
    projection = private_batch.run_private_batch(
        private_batch.prepare_private_batch(repository, private_root, workspace_root),
        register=True,
    )
    assert projection.status == "FAILED"
    assert "case_0100" not in projection.model_dump_json()
    evaluator = native_case[1].family.corpus_store("evaluator")
    children = evaluator.children(projection.batch_id)
    assert len(children) == 1
    assert children[0].status == "FAILED"
    assert native_case[1].family.ledger.committed_through() == []
    assert _public_tree(public_store) == before


@pytest.mark.parametrize("attack", ["mode", "hardlink", "symlink", "hash", "overlap"])
def test_private_preflight_rejects_unsafe_or_changed_inputs_without_runs(
    native_case, tmp_path, attack
):
    from gpu_agent.benchmark.private_batch import PrivateBatchInputError, prepare_private_batch

    repository, private_root, workspace_root, _ = _private_fixture(native_case, tmp_path)
    if attack == "mode":
        (private_root / "registry.json").chmod(0o644)
    elif attack == "hardlink":
        os.link(private_root / "input.json", private_root / "input-alias.json")
    elif attack == "symlink":
        target = private_root / "input.json"
        target.rename(private_root / "real-input.json")
        target.symlink_to(private_root / "real-input.json")
    elif attack == "hash":
        (private_root / "clean/kernel.cu").write_bytes(b"changed")
    else:
        nested = repository / "private"
        private_root.rename(nested)
        private_root = nested
    evaluator = native_case[1].family.corpus_store("evaluator")
    with pytest.raises(PrivateBatchInputError):
        prepare_private_batch(repository, private_root, workspace_root)
    assert [path for path in evaluator.root.iterdir() if len(path.name) == 32] == []


def test_private_batch_has_no_cli_or_public_export_surface():
    import gpu_agent.cli as cli

    source = Path(cli.__file__).read_text()
    assert "private-batch" not in source
    assert "run_private_batch" not in source


def test_private_batch_rejects_post_preflight_change_before_creating_parent(
    native_case, tmp_path, monkeypatch
):
    from gpu_agent.benchmark import private_batch

    repository, private_root, workspace_root, _ = _private_fixture(native_case, tmp_path)
    _use_simulated_backend(monkeypatch, private_batch, native_case)
    prepared = private_batch.prepare_private_batch(repository, private_root, workspace_root)
    source = private_root / "clean/kernel.cu"
    source.write_bytes(source.read_bytes() + b"\n// changed after preflight\n")
    source.chmod(0o600)

    with pytest.raises(private_batch.PrivateBatchInputError) as caught:
        private_batch.run_private_batch(prepared)
    assert caught.value.code in {"PRIVATE_INPUT_INVALID", "PRIVATE_PREFLIGHT_CHANGED"}
    evaluator = native_case[1].family.corpus_store("evaluator")
    assert [path for path in evaluator.root.iterdir() if len(path.name) == 32] == []


def test_evaluator_child_requires_running_private_batch_parent(native_case):
    from gpu_agent.benchmark.batch_security import create_seed_child

    controller = native_case[1]
    evaluator = controller.family.corpus_store("evaluator")
    wrong_parent = evaluator.create_run("seed_batch", binding=controller.binding)
    evaluator.transition(wrong_parent.id, "RUNNING", "VERIFYING")

    with pytest.raises(ValueError, match="RUNNING private_seed_batch"):
        create_seed_child(evaluator, wrong_parent.id, controller.binding)
    assert evaluator.children(wrong_parent.id) == []

    finalizing = evaluator.create_run("private_seed_batch", binding=controller.binding)
    evaluator.transition(finalizing.id, "RUNNING", "FINALIZING")
    with pytest.raises(ValueError, match="RUNNING private_seed_batch"):
        create_seed_child(evaluator, finalizing.id, controller.binding)
    assert evaluator.children(finalizing.id) == []


def test_runtime_private_input_change_returns_opaque_failure_and_terminalizes_runs(
    native_case, tmp_path, monkeypatch
):
    from gpu_agent.benchmark import private_batch

    repository, private_root, workspace_root, _ = _private_fixture(native_case, tmp_path)
    base = type(native_case[1].backend)

    class MutatingBackend(base):
        changed = False

        def _container(self, path, operation, timeout, *, stdin=b"", cancel=None):
            result = super()._container(path, operation, timeout, stdin=stdin, cancel=cancel)
            if operation == "build" and not self.changed:
                (private_root / "registry.json").chmod(0o400)
                self.changed = True
            return result

    _use_simulated_backend(monkeypatch, private_batch, native_case, MutatingBackend)
    projection = private_batch.run_private_batch(
        private_batch.prepare_private_batch(repository, private_root, workspace_root)
    )

    assert projection.status == "FAILED"
    assert set(projection.model_dump()) == {
        "batch_id",
        "case_count",
        "status",
        "private_batch_hash",
    }
    evaluator = native_case[1].family.corpus_store("evaluator")
    assert evaluator.load(projection.batch_id).status == "FAILED"
    children = evaluator.children(projection.batch_id)
    assert len(children) == 1
    assert children[0].status == "FAILED"
