"""Controller-owned execution and persistence of final release test evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field

from gpu_agent.benchmark.release import ReleaseTestEvidence, TestCounts
from gpu_agent.contracts import CurrentPhase, RepositorySnapshot, RunBinding, RunStatus
from gpu_agent.execution.models import ExecutionModel
from gpu_agent.execution.process import ProcessCapture, ProcessExecutor
from gpu_agent.provenance import capture_repository_snapshot
from gpu_agent.store import RunStore, read_regular

_TIMEOUT_SECONDS = 3600.0
_LOG_LIMIT = 4 * 1024 * 1024
_REPORT_LIMIT = 4 * 1024 * 1024
_JUNIT_LIMIT = 16 * 1024 * 1024
_ALLOWLIST_LIMIT = 1024 * 1024
_NODE_ID = re.compile(r"^[A-Za-z0-9_./:\[\],=+\-]+$")


class ReleaseProcess(Protocol):
    def execute(
        self,
        argv: list[str],
        cwd: Path,
        timeout_seconds: float,
        max_log_bytes: int,
        *,
        env: dict[str, str] | None = None,
    ) -> ProcessCapture: ...


class ReleaseNodeRequirement(ExecutionModel):
    node_id: str = Field(min_length=1, max_length=512)
    required_markers: list[str] = Field(min_length=1)


class ReleaseTestAllowlist(ExecutionModel):
    schema_version: Literal[1] = 1
    nodes: list[ReleaseNodeRequirement] = Field(min_length=1)

    def validate_unique(self) -> None:
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)) or ids != sorted(ids):
            raise ValueError("release test allowlist must be unique and sorted")
        for node in self.nodes:
            if _NODE_ID.fullmatch(node.node_id) is None:
                raise ValueError("release test allowlist contains an unsafe node ID")
            if "release_evidence" not in node.required_markers or len(node.required_markers) != len(
                set(node.required_markers)
            ):
                raise ValueError("release test marker requirements are invalid")


class PytestEvidenceReport(ExecutionModel):
    schema_version: Literal[1] = 1
    collected_node_ids: list[str]
    markers: dict[str, list[str]]
    passed_node_ids: list[str]
    skipped_node_ids: list[str]
    failed_node_ids: list[str]
    exit_status: int


class ReleaseTestInvocation(ExecutionModel):
    schema_version: Literal[1] = 1
    repository: RepositorySnapshot
    argv: list[str]
    allowlist_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    report_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    junit_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    stdout_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    stderr_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    exit_code: int | None
    timed_out: bool
    cancelled: bool
    truncated: bool
    tool_error: str | None


class ReleaseEvidenceController:
    """Run a fixed pytest selection and store same-commit, fail-closed evidence."""

    def __init__(
        self,
        store: RunStore,
        repository: Path,
        binding: RunBinding,
        *,
        process: ReleaseProcess | None = None,
        snapshot_capture: Callable[..., RepositorySnapshot] = capture_repository_snapshot,
    ) -> None:
        if store.visibility != "public" or binding.purpose != "release_acceptance":
            raise ValueError("release evidence requires a public release-acceptance binding")
        if binding.toolchain_lock_hash is None or binding.model_config_hash is None:
            raise ValueError("release evidence binding is incomplete")
        self.store = store
        self.repository = repository.absolute()
        self.binding = binding
        self.toolchain_hash = binding.toolchain_lock_hash
        self.model_config_hash = binding.model_config_hash
        self.process = process or ProcessExecutor()
        self.snapshot_capture = snapshot_capture

    def collect(self, corpus_cutoff: int) -> str:
        if corpus_cutoff < 1:
            raise ValueError("release evidence requires a positive corpus cutoff")
        before = self.snapshot_capture(
            self.repository, expected_commit=self.binding.repository.commit
        )
        if before != self.binding.repository:
            raise ValueError("release binding differs from the repository snapshot")
        allowlist_path = self.repository / "evaluation/release-test-allowlist.json"
        allowlist_bytes = read_regular(allowlist_path, _ALLOWLIST_LIMIT)
        allowlist = ReleaseTestAllowlist.model_validate_json(allowlist_bytes)
        allowlist.validate_unique()

        run = self.store.create_run("release_test", binding=self.binding)
        self.store.transition(run.id, RunStatus.RUNNING, CurrentPhase.EXECUTING)
        try:
            with tempfile.TemporaryDirectory(prefix="gpu-agent-release-") as temporary:
                root = Path(temporary)
                report_path = root / "pytest-evidence.json"
                junit_path = root / "pytest-junit.xml"
                argv = [
                    sys.executable,
                    "-I",
                    "-m",
                    "pytest",
                    "-m",
                    "release_evidence",
                    "--require-live",
                    f"--release-evidence-report={report_path}",
                    f"--junitxml={junit_path}",
                    "-q",
                ]
                environment = dict(os.environ)
                environment.update(
                    {
                        "LC_ALL": "C.UTF-8",
                        "PYTHONHASHSEED": "0",
                        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                    }
                )
                capture = self.process.execute(
                    argv,
                    self.repository,
                    _TIMEOUT_SECONDS,
                    _LOG_LIMIT,
                    env=environment,
                )
                report_bytes = read_regular(report_path, _REPORT_LIMIT)
                junit_bytes = read_regular(junit_path, _JUNIT_LIMIT)
                report = PytestEvidenceReport.model_validate_json(report_bytes)
                invocation = ReleaseTestInvocation(
                    repository=before,
                    argv=self._normalized_argv(argv),
                    allowlist_hash=_sha256(allowlist_bytes),
                    report_hash=_sha256(report_bytes),
                    junit_hash=_sha256(junit_bytes),
                    stdout_hash=_sha256(capture.stdout),
                    stderr_hash=_sha256(capture.stderr),
                    exit_code=capture.exit_code,
                    timed_out=capture.timed_out,
                    cancelled=capture.cancelled,
                    truncated=capture.truncated,
                    tool_error=capture.tool_error,
                )
                self.store.put(
                    run.id,
                    "release/test-invocation.json",
                    invocation.model_dump_json().encode(),
                    "public",
                )
                self._validate(capture, report, allowlist, junit_bytes)
                after = self.snapshot_capture(
                    self.repository, expected_commit=self.binding.repository.commit
                )
                if after != before:
                    raise ValueError("repository changed during release evidence collection")
                node_ids = sorted(report.collected_node_ids)
                evidence = ReleaseTestEvidence(
                    run_id=run.id,
                    repository=before,
                    toolchain_hash=self.toolchain_hash,
                    model_config_hash=self.model_config_hash,
                    corpus_cutoff=corpus_cutoff,
                    test_counts=TestCounts(
                        expected=len(allowlist.nodes),
                        executed=len(report.passed_node_ids),
                        skipped_required=len(report.skipped_node_ids),
                        failed=len(report.failed_node_ids),
                    ),
                    collected_node_ids=node_ids,
                    collection_hash=_sha256(json.dumps(node_ids, separators=(",", ":")).encode()),
                    exit_code=0,
                )
                self.store.put(
                    run.id,
                    "release/test-evidence.json",
                    evidence.model_dump_json().encode(),
                    "public",
                )
            self.store.transition(run.id, RunStatus.RUNNING, CurrentPhase.FINALIZING)
            self.store.transition(run.id, RunStatus.COMPLETED, None)
            return run.id
        except (OSError, ValueError):
            current = self.store.load(run.id)
            if current.status == RunStatus.RUNNING:
                self.store.transition(run.id, RunStatus.FAILED, None)
            raise

    @staticmethod
    def _normalized_argv(argv: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in argv:
            if value.startswith("--release-evidence-report="):
                normalized.append("--release-evidence-report=<controller-temp>")
            elif value.startswith("--junitxml="):
                normalized.append("--junitxml=<controller-temp>")
            else:
                normalized.append(value)
        return normalized

    @staticmethod
    def _validate(
        capture: ProcessCapture,
        report: PytestEvidenceReport,
        allowlist: ReleaseTestAllowlist,
        junit_bytes: bytes,
    ) -> None:
        if (
            capture.exit_code != 0
            or capture.timed_out
            or capture.cancelled
            or capture.truncated
            or capture.tool_error is not None
            or report.exit_status != 0
            or report.skipped_node_ids
            or report.failed_node_ids
        ):
            raise ValueError("required release test invocation did not pass cleanly")
        required = {node.node_id: set(node.required_markers) for node in allowlist.nodes}
        collected = report.collected_node_ids
        if (
            collected != sorted(collected)
            or len(collected) != len(set(collected))
            or set(collected) != set(required)
            or set(report.passed_node_ids) != set(required)
            or set(report.markers) != set(required)
        ):
            raise ValueError("release test collection differs from the frozen allowlist")
        for node_id, markers in report.markers.items():
            if not required[node_id].issubset(set(markers)):
                raise ValueError("release test marker metadata differs from the allowlist")
        if (
            b"<testsuite" not in junit_bytes
            or b"<failure" in junit_bytes
            or b"<skipped" in junit_bytes
        ):
            raise ValueError("release JUnit report is incomplete")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
