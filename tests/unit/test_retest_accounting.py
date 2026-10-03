import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from gpu_agent.agent.provider import Invocation, Usage
from gpu_agent.contracts import now

spec = importlib.util.spec_from_file_location(
    "retest_accounting", Path(__file__).resolve().parents[2] / "tools/retest_accounting.py"
)
assert spec and spec.loader
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def fixture(tmp_path):
    rid = "a" * 32
    root = tmp_path / "report"
    artifacts = root / "public" / rid / "artifacts"
    artifacts.mkdir(parents=True)
    call = Invocation(
        invocation_id="b" * 32,
        run_id=rid,
        kind="plan",
        attempt=0,
        state="STARTED",
        started_at=now(),
        configured_model="test",
        endpoint_host="example.org",
        client_request_id="c" * 32,
    )
    refs = []
    for i, item in enumerate(
        [
            call,
            call.model_copy(
                update={"state": "COMPLETED", "usage": Usage(input_tokens=10, output_tokens=20)}
            ),
        ]
    ):
        data = item.model_dump_json().encode()
        (artifacts / str(i)).write_bytes(data)
        refs.append(
            {
                "name": f"provider/{item.invocation_id}/{item.state}.json",
                "visibility": "public",
                "run_id": rid,
                "relative_path": f"{rid}/artifacts/{i}",
                "byte_count": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    (artifacts.parent / "manifest.json").write_text(
        json.dumps({"id": rid, "kind": "diagnosis", "artifact_refs": refs})
    )
    (root / "results.jsonl").write_text(json.dumps({"run_id": rid, "physical_calls": 1}) + "\n")
    return root, artifacts


def test_read_only_one_call_and_unknown_rate(tmp_path):
    root, artifacts = fixture(tmp_path)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = tool.report(root, None)
    assert result["total"]["physical_calls"] == 1
    assert result["total"]["estimated_cost_usd"] is None
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    (artifacts / "1").write_text("tampered")
    with pytest.raises(ValueError, match="hash or size"):
        tool.report(root, None)


def test_symlink_refused(tmp_path):
    root, artifacts = fixture(tmp_path)
    target = tmp_path / "target"
    (artifacts / "1").rename(target)
    (artifacts / "1").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        tool.report(root, None)
