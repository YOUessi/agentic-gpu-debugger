"""Controller-owned execution and persistence of final release test evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

from pydantic import Field

from gpu_agent.benchmark.release import ReleaseTestEvidence, TestCounts
from gpu_agent.contracts import (
    ArtifactRef,
    CurrentPhase,
    RepositorySnapshot,
    RunBinding,
    RunManifest,
    RunStatus,
)
from gpu_agent.execution.models import ExecutionModel
from gpu_agent.execution.process import ProcessCapture, ProcessExecutor
from gpu_agent.provenance import capture_repository_snapshot
from gpu_agent.store import RunStore, read_regular

_TIMEOUT_SECONDS = 3600.0
_LOG_LIMIT = 4 * 1024 * 1024
_REPORT_LIMIT = 4 * 1024 * 1024
_JUNIT_LIMIT = 16 * 1024 * 1024
_ALLOWLIST_LIMIT = 1024 * 1024
_ARCHIVE_LIMIT = 256 * 1024 * 1024
_NODE_ID = re.compile(r"^[A-Za-z0-9_./:\[\],=+\-]+$")
_GIT = "/usr/bin/git"
_RAW_ARTIFACTS = {
    "allowlist": "release/test-allowlist.json",
    "report": "release/pytest-evidence.json",
    "junit": "release/pytest-junit.xml",
    "stdout": "release/pytest-stdout.log",
    "stderr": "release/pytest-stderr.log",
}
RELEASE_TEST_INVOCATION_ARTIFACT = "release/test-invocation.json"
RELEASE_TEST_EVIDENCE_ARTIFACT = "release/test-evidence.json"
RELEASE_TEST_ARTIFACT_NAMES = frozenset(
    {
        *_RAW_ARTIFACTS.values(),
        RELEASE_TEST_INVOCATION_ARTIFACT,
        RELEASE_TEST_EVIDENCE_ARTIFACT,
    }
)
_NORMALIZED_ARGV = [
    "<python-executable>",
    "-I",
    "-c",
    "<controller-bootstrap>",
    "-m",
    "release_evidence",
    "--require-live",
    "--release-evidence-report=<controller-temp>",
    "--junitxml=<controller-temp>",
    "-q",
]


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
            markers = node.required_markers
            if (
                "release_evidence" not in markers
                or len(markers) != len(set(markers))
                or markers != sorted(markers)
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
    gpu_agent_origin: str = Field(min_length=1, max_length=4096)


class ReleaseTestInvocation(ExecutionModel):
    schema_version: Literal[2] = 2
    repository: RepositorySnapshot
    argv: list[str]
    source_archive_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_module_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
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


SourceSnapshotFactory = Callable[[Path, RepositorySnapshot], tuple[Path, str]]


class ReleaseEvidenceController:
    """Run a fixed suite from the exact commit archive and retain auditable evidence."""

    def __init__(
        self,
        store: RunStore,
        repository: Path,
        binding: RunBinding,
        *,
        process: ReleaseProcess | None = None,
        snapshot_capture: Callable[..., RepositorySnapshot] = capture_repository_snapshot,
        source_snapshot: SourceSnapshotFactory | None = None,
    ) -> None:
        if store.visibility != "public" or binding.purpose != "release_acceptance":
            raise ValueError("release evidence requires a public release-acceptance binding")
        if (
            binding.toolchain_lock_hash is None
            or binding.model_config_hash is None
            or binding.prompt_version is None
            or binding.corpus_ledger_namespace_hash is None
        ):
            raise ValueError("release evidence binding is incomplete")
        self.store = store
        self.repository = repository.absolute()
        self.binding = binding
        self.toolchain_hash = binding.toolchain_lock_hash
        self.model_config_hash = binding.model_config_hash
        self.prompt_version = binding.prompt_version
        self.process = process or ProcessExecutor()
        self.snapshot_capture = snapshot_capture
        self.source_snapshot = source_snapshot or self._archive_snapshot

    def collect(self, corpus_cutoff: int) -> str:
        if corpus_cutoff < 1:
            raise ValueError("release evidence requires a positive corpus cutoff")
        before = self.snapshot_capture(
            self.repository, expected_commit=self.binding.repository.commit
        )
        if before != self.binding.repository:
            raise ValueError("release binding differs from the repository snapshot")

        run = self.store.create_run("release_test", binding=self.binding)
        self.store.transition(run.id, RunStatus.RUNNING, CurrentPhase.EXECUTING)
        try:
            with tempfile.TemporaryDirectory(prefix="gpu-agent-release-") as temporary:
                root = Path(temporary)
                source_root, archive_hash = self.source_snapshot(root, before)
                allowlist_bytes = read_regular(
                    source_root / "evaluation/release-test-allowlist.json",
                    _ALLOWLIST_LIMIT,
                )
                allowlist = ReleaseTestAllowlist.model_validate_json(allowlist_bytes)
                allowlist.validate_unique()
                report_path = root / "pytest-evidence.json"
                junit_path = root / "pytest-junit.xml"
                bootstrap = (
                    "import sys;"
                    f"sys.path.insert(0,{str(source_root / 'src')!r});"
                    "import pytest;"
                    "raise SystemExit(pytest.main(sys.argv[1:]))"
                )
                argv = [
                    sys.executable,
                    "-I",
                    "-c",
                    bootstrap,
                    "-m",
                    "release_evidence",
                    "--require-live",
                    f"--release-evidence-report={report_path}",
                    f"--junitxml={junit_path}",
                    "-q",
                ]
                capture = self.process.execute(
                    argv,
                    source_root,
                    _TIMEOUT_SECONDS,
                    _LOG_LIMIT,
                    env=self._environment(),
                )
                report_bytes = read_regular(report_path, _REPORT_LIMIT)
                junit_bytes = read_regular(junit_path, _JUNIT_LIMIT)
                report = PytestEvidenceReport.model_validate_json(report_bytes)
                expected_module = source_root / "src/gpu_agent/__init__.py"
                imported_module = Path(report.gpu_agent_origin).resolve(strict=True)
                if imported_module != expected_module.resolve(strict=True):
                    raise ValueError("release tests imported gpu_agent outside the commit snapshot")
                module_bytes = read_regular(imported_module, 1024 * 1024)
                raw = {
                    "allowlist": allowlist_bytes,
                    "report": report_bytes,
                    "junit": junit_bytes,
                    "stdout": capture.stdout,
                    "stderr": capture.stderr,
                }
                raw_refs = {
                    key: self.store.put(run.id, _RAW_ARTIFACTS[key], value, "public")
                    for key, value in raw.items()
                }
                invocation = ReleaseTestInvocation(
                    repository=before,
                    argv=list(_NORMALIZED_ARGV),
                    source_archive_hash=archive_hash,
                    source_module_hash=_sha256(module_bytes),
                    allowlist_hash=raw_refs["allowlist"].sha256,
                    report_hash=raw_refs["report"].sha256,
                    junit_hash=raw_refs["junit"].sha256,
                    stdout_hash=raw_refs["stdout"].sha256,
                    stderr_hash=raw_refs["stderr"].sha256,
                    exit_code=capture.exit_code,
                    timed_out=capture.timed_out,
                    cancelled=capture.cancelled,
                    truncated=capture.truncated,
                    tool_error=capture.tool_error,
                )
                invocation_ref = self.store.put(
                    run.id,
                    RELEASE_TEST_INVOCATION_ARTIFACT,
                    invocation.model_dump_json().encode(),
                    "public",
                )
                _validate(invocation, report, allowlist, junit_bytes)
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
                    prompt_version=self.prompt_version,
                    corpus_cutoff=corpus_cutoff,
                    test_counts=TestCounts(
                        expected=len(allowlist.nodes),
                        executed=len(report.passed_node_ids),
                        skipped_required=len(report.skipped_node_ids),
                        failed=len(report.failed_node_ids),
                    ),
                    collected_node_ids=node_ids,
                    collection_hash=_sha256(json.dumps(node_ids, separators=(",", ":")).encode()),
                    invocation_hash=invocation_ref.sha256,
                    exit_code=0,
                )
                self.store.put(
                    run.id,
                    RELEASE_TEST_EVIDENCE_ARTIFACT,
                    evidence.model_dump_json().encode(),
                    "public",
                )
            self.store.transition(run.id, RunStatus.RUNNING, CurrentPhase.FINALIZING)
            self.store.transition(run.id, RunStatus.COMPLETED, None)
            return run.id
        except BaseException:
            try:
                current = self.store.load(run.id)
                if current.status == RunStatus.RUNNING:
                    self.store.transition(run.id, RunStatus.FAILED, None)
            except BaseException:
                pass
            raise

    def _archive_snapshot(
        self, temporary_root: Path, snapshot: RepositorySnapshot
    ) -> tuple[Path, str]:
        archive = temporary_root / "source.tar"
        capture = self.process.execute(
            [
                _GIT,
                "-c",
                "core.fsmonitor=false",
                "archive",
                "--format=tar",
                f"--output={archive}",
                snapshot.commit,
            ],
            self.repository,
            60,
            1024 * 1024,
            env={
                "PATH": "/usr/bin:/bin",
                "LC_ALL": "C",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null",
            },
        )
        if (
            capture.exit_code != 0
            or capture.timed_out
            or capture.cancelled
            or capture.truncated
            or capture.tool_error is not None
            or capture.stdout
            or capture.stderr
        ):
            raise ValueError("repository archive creation failed")
        archive_bytes = read_regular(archive, _ARCHIVE_LIMIT)
        source = temporary_root / "source"
        source.mkdir(mode=0o700)
        seen: set[str] = set()
        total = 0
        with tarfile.open(archive, mode="r:") as stream:
            for member in stream.getmembers():
                path = PurePosixPath(member.name)
                if (
                    not member.name
                    or path.is_absolute()
                    or path.as_posix() != member.name.rstrip("/")
                    or ".." in path.parts
                    or ".git" in path.parts
                    or member.name in seen
                    or not (member.isdir() or member.isfile())
                ):
                    raise ValueError("repository archive contains an unsafe member")
                seen.add(member.name)
                target = source.joinpath(*path.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True, mode=0o700)
                    continue
                total += member.size
                if total > _ARCHIVE_LIMIT:
                    raise ValueError("repository archive expands beyond its bound")
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                extracted = stream.extractfile(member)
                if extracted is None:
                    raise ValueError("repository archive member is unreadable")
                content = extracted.read(_ARCHIVE_LIMIT + 1)
                if len(content) != member.size:
                    raise ValueError("repository archive member size changed")
                target.write_bytes(content)
                target.chmod(0o500 if member.mode & 0o111 else 0o400)
        for directory, children, _files in os.walk(source, topdown=False):
            Path(directory).chmod(0o500)
            for child in children:
                (Path(directory) / child).chmod(0o500)
        return source, _sha256(archive_bytes)

    @staticmethod
    def _environment() -> dict[str, str]:
        environment = {
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
        for name in (
            "HOME",
            "XDG_RUNTIME_DIR",
            "DOCKER_HOST",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "GPU_AGENT_CUDA_ROOT",
            "GPU_AGENT_CUDA_BIN",
            "LD_LIBRARY_PATH",
        ):
            value = os.environ.get(name)
            if value:
                environment[name] = value
        return environment


def verify_persisted_release_artifacts(store: RunStore, run: RunManifest) -> None:
    """Recompute release-test evidence from retained raw artifacts."""
    if run.kind != "release_test" or run.binding is None:
        raise ValueError("run is not release test evidence")
    raw_refs = {key: _one(run, name) for key, name in _RAW_ARTIFACTS.items()}
    invocation_ref = _one(run, RELEASE_TEST_INVOCATION_ARTIFACT)
    invocation = ReleaseTestInvocation.model_validate_json(store.read(invocation_ref))
    allowlist_bytes = store.read(raw_refs["allowlist"])
    report_bytes = store.read(raw_refs["report"])
    junit_bytes = store.read(raw_refs["junit"])
    allowlist = ReleaseTestAllowlist.model_validate_json(allowlist_bytes)
    report = PytestEvidenceReport.model_validate_json(report_bytes)
    if (
        invocation.repository != run.binding.repository
        or invocation.argv != _NORMALIZED_ARGV
        or invocation.allowlist_hash != raw_refs["allowlist"].sha256
        or invocation.report_hash != raw_refs["report"].sha256
        or invocation.junit_hash != raw_refs["junit"].sha256
        or invocation.stdout_hash != raw_refs["stdout"].sha256
        or invocation.stderr_hash != raw_refs["stderr"].sha256
    ):
        raise ValueError("release invocation differs from retained raw artifacts")
    _validate(invocation, report, allowlist, junit_bytes)


def _validate(
    invocation: ReleaseTestInvocation,
    report: PytestEvidenceReport,
    allowlist: ReleaseTestAllowlist,
    junit_bytes: bytes,
) -> None:
    allowlist.validate_unique()
    if (
        invocation.argv != _NORMALIZED_ARGV
        or invocation.exit_code != 0
        or invocation.timed_out
        or invocation.cancelled
        or invocation.truncated
        or invocation.tool_error is not None
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
        or len(report.passed_node_ids) != len(set(report.passed_node_ids))
        or set(report.markers) != set(required)
    ):
        raise ValueError("release test collection differs from the frozen allowlist")
    for node_id, markers in report.markers.items():
        if markers != sorted(set(markers)) or not required[node_id].issubset(set(markers)):
            raise ValueError("release test marker metadata differs from the allowlist")
    _validate_junit(junit_bytes, len(required))


def _validate_junit(content: bytes, expected: int) -> None:
    if b"<!DOCTYPE" in content.upper() or b"<!ENTITY" in content.upper():
        raise ValueError("release JUnit contains forbidden declarations")
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError("release JUnit is invalid") from exc
    cases = root.findall(".//testcase")
    if len(cases) != expected:
        raise ValueError("release JUnit test count differs from the allowlist")
    if any(
        case.find(name) is not None for case in cases for name in ("failure", "error", "skipped")
    ):
        raise ValueError("release JUnit contains a non-passing test")
    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    if not suites:
        raise ValueError("release JUnit has no test suite")
    for suite in suites:
        for name in ("failures", "errors", "skipped"):
            if int(suite.attrib.get(name, "0")) != 0:
                raise ValueError("release JUnit suite reports a non-passing test")


def _one(run: RunManifest, name: str) -> ArtifactRef:
    refs = [ref for ref in run.artifact_refs if ref.name == name]
    if len(refs) != 1:
        raise ValueError(f"expected exactly one {name} artifact")
    return refs[0]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
