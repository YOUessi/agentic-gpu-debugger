import hashlib
import json

import pytest

from gpu_agent.public_task import PublicTask


@pytest.mark.parametrize(
    "fault", ["missing", "version", "algorithm", "hash", "symlink", "shape", "nan", "overflow"]
)
def test_bad_repair_input_rejected_before_run_or_external_work(oob_service, monkeypatch, fault):
    service, provider, source = oob_service
    value = PublicTask(
        source_sha256=hashlib.sha256((source / "kernel.cu").read_bytes()).hexdigest(),
        algorithm="vector-add-cpu-v1",
    ).model_dump()
    if fault in {"version", "algorithm"}:
        value[fault] = "unsupported"
    if fault == "hash":
        value["source_sha256"] = "0" * 64
    if fault != "missing":
        (source / "task.json").write_text(json.dumps(value))
    if fault == "symlink":
        (source / "task.json").rename(source / "other.json")
        (source / "task.json").symlink_to(source / "other.json")
    if fault in {"shape", "nan", "overflow"}:
        a = {"shape": [], "nan": [float("nan")], "overflow": [1e100]}[fault]
        (source / "input.json").write_text(json.dumps({"n": 1, "a": a, "b": [2]}))

    def forbidden(*args, **kwargs):
        pytest.fail("preflight must precede run creation and external execution")

    monkeypatch.setattr(service.store, "create_run", forbidden)
    monkeypatch.setattr(service, "_backend_factory", forbidden)
    with pytest.raises(ValueError):
        service.repair(source)
    assert not provider.kinds


def test_cli_reports_missing_public_task_without_external_work(oob_service, monkeypatch):
    from typer.testing import CliRunner

    from gpu_agent.cli import app
    from gpu_agent.service import ApplicationService

    service, provider, source = oob_service
    monkeypatch.setattr(ApplicationService, "configured", lambda: service)
    result = CliRunner().invoke(app, ["repair", str(source)])
    assert result.exit_code == 2
    assert "PUBLIC_TASK_UNAVAILABLE" in result.output
    assert not provider.kinds


def test_repair_rejects_unsupported_interface_before_run(oob_service, monkeypatch):
    service, provider, source = oob_service
    data = b"int main() { return 0; }\n"
    (source / "kernel.cu").write_bytes(data)
    (source / "task.json").write_text(
        PublicTask(
            source_sha256=hashlib.sha256(data).hexdigest(), algorithm="vector-add-cpu-v1"
        ).model_dump_json()
    )
    monkeypatch.setattr(service.store, "create_run", lambda *a, **kw: pytest.fail("too late"))
    with pytest.raises(ValueError, match="PUBLIC_INTERFACE_UNSUPPORTED"):
        service.repair(source)
    assert not provider.kinds
