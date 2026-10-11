"""Read-only summaries and bounded, public-only diagnostic exports."""

import hashlib
import json
import zipfile
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path, PurePosixPath

from gpu_agent.benchmark.batch_models import BatchSummary, RoleSummary
from gpu_agent.benchmark.models import CaseExecutionPlan, CaseOracleObservation
from gpu_agent.benchmark.validation import derive_oracle
from gpu_agent.contracts import ArtifactRef, RunManifest, RunStatus
from gpu_agent.evidence.sanitizer import parse_sanitizer
from gpu_agent.execution.models import BuildResult, ExecutionResult, SanitizerResult
from gpu_agent.execution.process import ProcessCapture
from gpu_agent.store import EvaluationRunLease, RunStore, reject_symlinks

EXPORT_LIMIT = 256 * 1024 * 1024


def _ref(run: RunManifest, name: str) -> ArtifactRef | None:
    matches = [ref for ref in run.artifact_refs if ref.name == name]
    if len(matches) > 1:
        raise ValueError("ambiguous named batch artifact")
    return matches[0] if matches else None


def summarize_role(store: RunStore, run_id: str | None, plan: CaseExecutionPlan) -> RoleSummary:
    """Classify observed failure stages, not an alternative validation/registration gate."""
    if store.visibility != "public" or plan.split != "public":
        raise ValueError("batch summaries cannot expose evaluator evidence")
    result = RoleSummary(run_id=run_id)

    def read_current(ref: ArtifactRef) -> bytes:
        if ref.visibility != "public" or ref.run_id != run_id:
            raise ValueError("batch evidence crosses run or visibility boundary")
        return store.read(ref)

    def fail(stage: str, reason: str) -> None:
        if result.reason_code is None:
            result.failure_stage, result.reason_code = stage, reason

    def process_error(stage: str, tool: object) -> None:
        for field, suffix in (
            ("cancelled", "CANCELLED"),
            ("timed_out", "TIMEOUT"),
            ("tool_error", "TOOL_ERROR"),
            ("truncated", "TRUNCATED"),
        ):
            if getattr(tool, field, False):
                fail(stage, f"{stage}_{suffix}")
                return

    if run_id is None:
        fail("PREPARING", "EXECUTION_NOT_STARTED")
        return result
    run = store.load(run_id)
    if run.kind != "case_execution" or run.external_origin is not None:
        raise ValueError("not a public native case execution")
    result.run_status = run.status
    build_ref = _ref(run, "validation/build-result.json")
    if build_ref is None:
        fail("PREPARING", "EXECUTION_FAILED")
        return result
    build = BuildResult.model_validate_json(read_current(build_ref))
    result.build_success = bool(
        build.binary_ref is not None
        and build.tool_result.exit_code == 0
        and not (
            build.tool_result.tool_error
            or build.tool_result.timed_out
            or build.tool_result.cancelled
            or build.tool_result.truncated
        )
    )
    process_error("BUILD", build.tool_result)
    if not result.build_success:
        fail("BUILD", "BUILD_FAILED")
        return result
    runtime_ref = _ref(run, "validation/runtime-result.json")
    if runtime_ref is None:
        fail("RUNTIME", "RUNTIME_RESULT_MISSING")
        return result
    runtime = ExecutionResult.model_validate_json(read_current(runtime_ref))
    result.runtime_status = "SUCCESS" if runtime.tool_result.exit_code == 0 else "FAILED"
    for flag, status in (
        ("cancelled", "CANCELLED"),
        ("timed_out", "TIMEOUT"),
        ("tool_error", "TOOL_ERROR"),
        ("truncated", "TRUNCATED"),
    ):
        if getattr(runtime.tool_result, flag):
            result.runtime_status = status
            break
    process_error("RUNTIME", runtime.tool_result)
    if plan.role == "clean" and result.runtime_status != "SUCCESS":
        fail("RUNTIME", "CLEAN_RUNTIME_FAILED")
    oracle_ref = _ref(run, "validation/oracle-result.json")
    if oracle_ref is None:
        fail("ORACLE", "ORACLE_RESULT_MISSING")
    else:
        oracle = CaseOracleObservation.model_validate_json(read_current(oracle_ref))
        derived_oracle = derive_oracle(
            read_current(runtime.tool_result.typed_payload.stdin_ref),
            read_current(runtime.output_ref),
            plan.oracle_id,
        )
        result.oracle_passed = derived_oracle.passed
        if (
            oracle.oracle_id != plan.oracle_id
            or oracle.channel != "ordinary"
            or oracle.input_ref != runtime.tool_result.typed_payload.stdin_ref
            or oracle.output_ref != runtime.output_ref
            or oracle.sanitizer_result_ref is not None
            or oracle.result != derived_oracle
        ):
            fail("EVIDENCE", "ORACLE_BINDING_MISMATCH")
        if plan.role == "clean" and not result.oracle_passed:
            fail("ORACLE", "CLEAN_ORACLE_FAILED")
    for index in range(plan.sanitizer_repetitions):
        ref = _ref(run, f"validation/sanitizer-{index:02d}.json")
        if ref is None:
            fail("SANITIZER", "SANITIZER_RESULT_MISSING")
            break
        check = SanitizerResult.model_validate_json(read_current(ref))
        observed = check
        if check.tool_result is not None:
            tool = check.tool_result
            observed = parse_sanitizer(
                plan.target_tool,
                ProcessCapture(
                    exit_code=tool.exit_code,
                    stdout=read_current(tool.stdout_artifact),
                    stderr=read_current(tool.stderr_artifact),
                    timed_out=tool.timed_out,
                    cancelled=tool.cancelled,
                    truncated=tool.truncated,
                    tool_error=tool.tool_error,
                ),
            )
            native_findings = [
                finding.model_copy(update={"raw_ref": None}) for finding in check.findings
            ]
            if not (
                tool.tool_name == "sanitizer"
                and tool.stdout_artifact.name == f"sanitizer/{tool.request_id}/program.stdout"
                and tool.stderr_artifact.name
                == f"sanitizer/{tool.request_id}/{plan.target_tool.value}.log"
                and tool.typed_payload.tool == plan.target_tool
                and tool.typed_payload.stdin_ref == runtime.tool_result.typed_payload.stdin_ref
                and tool.typed_payload.binary_ref == build.binary_ref
                and tool.typed_payload.program_output_ref == check.program_output_ref
                and check.program_output_ref == tool.stdout_artifact
                and tool.typed_payload.program_stderr_ref is not None
                and tool.typed_payload.program_stderr_ref.name
                == f"sanitizer/{tool.request_id}/program.stderr"
                and check.status == tool.typed_payload.status == observed.status
                and check.parser_version
                == tool.typed_payload.parser_version
                == observed.parser_version
                and check.check_outcome == tool.typed_payload.check_outcome
                and check.findings == tool.typed_payload.findings
                and check.completed == tool.typed_payload.completed
                and check.check_outcome == observed.check_outcome
                and check.completed == observed.completed
                and native_findings == observed.findings
            ):
                fail("EVIDENCE", "SANITIZER_BINDING_MISMATCH")
            process_error("SANITIZER", tool)
        else:
            fail("SANITIZER", "SANITIZER_EVIDENCE_MISSING")
        result.sanitizer_outcomes.append(observed.check_outcome)
        detected = any(f.category == plan.expected_finding for f in observed.findings)
        result.target_detections.append(detected)
        if observed.check_outcome in {"TOOL_ERROR", "UNSUPPORTED"}:
            fail("SANITIZER", f"SANITIZER_{observed.check_outcome}")
        elif not observed.completed:
            fail("SANITIZER", "SANITIZER_INCOMPLETE")
        elif plan.role == "clean" and observed.check_outcome != "CLEAN":
            fail("SANITIZER", "CLEAN_SANITIZER_NOT_CLEAN")
        elif plan.role == "mutant" and not detected:
            fail("SANITIZER", "TARGET_FINDING_MISSING")
        oracle_ref = _ref(run, f"validation/sanitizer-oracle-{index:02d}.json")
        if oracle_ref is not None:
            instrumented = CaseOracleObservation.model_validate_json(read_current(oracle_ref))
            if check.program_output_ref is None:
                fail("ORACLE", "INSTRUMENTED_OUTPUT_MISSING")
                continue
            passed = derive_oracle(
                read_current(runtime.tool_result.typed_payload.stdin_ref),
                read_current(check.program_output_ref),
                plan.oracle_id,
            )
            if (
                instrumented.oracle_id != plan.oracle_id
                or instrumented.channel != "instrumented"
                or instrumented.input_ref != runtime.tool_result.typed_payload.stdin_ref
                or instrumented.output_ref != check.program_output_ref
                or instrumented.sanitizer_result_ref != ref
                or instrumented.result != passed
            ):
                fail("EVIDENCE", "INSTRUMENTED_ORACLE_BINDING_MISMATCH")
            result.instrumented_oracles.append(passed.passed)
            if plan.role == "clean" and not passed.passed:
                fail("ORACLE", "CLEAN_INSTRUMENTED_ORACLE_FAILED")
        else:
            fail("ORACLE", "INSTRUMENTED_ORACLE_MISSING")
    if _ref(run, "validation/cleanup-error.json") is not None:
        fail("CLEANUP", "WORKSPACE_CLEANUP_FAILED")
    if run.status != RunStatus.COMPLETED:
        fail("EXECUTION", "EXECUTION_INCOMPLETE")
    return result


def _summary_from_manifest(
    run: RunManifest, batch_id: str, read: Callable[[ArtifactRef], bytes]
) -> BatchSummary:
    if run.kind != "seed_batch":
        raise ValueError("not a seed batch run")
    ref = _ref(run, "batch/summary.json")
    if ref is None:
        progress = sorted(
            (r for r in run.artifact_refs if r.name.startswith("batch/progress/")),
            key=lambda r: r.name,
        )
        if not progress:
            raise ValueError("batch has no saved progress")
        ref = progress[-1]
    summary = BatchSummary.model_validate_json(read(ref))
    if summary.batch_run_id != batch_id:
        raise ValueError("batch summary identity mismatch")
    return summary


def load_batch_summary(store: RunStore, batch_id: str) -> BatchSummary:
    if store.visibility != "public":
        raise ValueError("batch reports support the public store only")
    return _summary_from_manifest(store.load(batch_id), batch_id, store.read)


def render_batch(summary: BatchSummary) -> str:
    lines = [
        "# GPU 种子案例批量验证报告",
        "",
        f"批次：`{summary.batch_run_id}`",
        f"流程状态：`{summary.status}`",
        f"请求注册：`{summary.register_requested}`",
        "",
        "本报告是操作摘要，不代表正式五模式评测或 Release 已通过。",
        "`COMPLETED` 仅表示批次处理结束；请逐项查看案例状态。",
        "",
        "| Case | 工具 | 状态 | 原因 | clean run | mutant run |",
        "|---|---|---|---|---|---|",
    ]
    for case in summary.cases:
        clean = case.clean.run_id if case.clean else None
        mutant = case.mutant.run_id if case.mutant else None
        lines.append(
            f"| {case.case_id} | {case.target_tool} | {case.status} | "
            f"{case.reason_code or '-'} | {clean or '-'} | {mutant or '-'} |"
        )
    lines.extend(["", "## 检查明细", ""])
    for case in summary.cases:
        lines.append(f"### {case.case_id}（每个版本计划 {case.repetitions} 次 Sanitizer）")
        for name in ("clean", "mutant"):
            role = getattr(case, name)
            if role is None:
                lines.append(f"{name}：未运行。")
                continue
            lines.append(
                f"{name}：build={role.build_success}，runtime={role.runtime_status}，"
                f"ordinary oracle={role.oracle_passed}，"
                f"sanitizer={role.sanitizer_outcomes}，"
                f"target finding={role.target_detections}，"
                f"instrumented oracle={role.instrumented_oracles}。"
            )
        lines.append("")
    if summary.stopped_reason:
        lines.append(f"停止原因：`{summary.stopped_reason}`")
    lines.extend(
        [
            "",
            "失败案例不会计入 corpus；VALIDATED 不等于 REGISTERED。",
            "每个案例只执行注册规格的目标 Sanitizer，并非修复后的四工具 strict verification。",
            "测试环境的模拟容器输出不构成真实 GPU 通过证据。",
            "",
        ]
    )
    return "\n".join(lines)


def _selected_runs(
    batch: RunManifest,
    summary: BatchSummary,
    leases: dict[str, EvaluationRunLease],
) -> list[RunManifest]:
    """Select exactly summary-named roles. Never enumerate arbitrary direct children."""
    if batch.binding is None or batch.binding.purpose != "corpus_validation":
        raise ValueError("batch is missing its native validation binding")
    runs = [batch]
    seen = {batch.id}
    cases = set()
    for case in summary.cases:
        if case.case_id in cases:
            raise ValueError("duplicate case in batch summary")
        cases.add(case.case_id)
        for role_name, role in (("clean", case.clean), ("mutant", case.mutant)):
            if role is None or role.run_id is None:
                continue
            if role.run_id in seen:
                raise ValueError("duplicate run in batch summary")
            seen.add(role.run_id)
            lease = leases.get(role.run_id)
            if lease is None:
                raise ValueError("summary child has no pinned lease")
            child = lease.load()
            if (
                child.parent_run_id != batch.id
                or child.kind != "case_execution"
                or child.binding != batch.binding
                or child.external_origin is not None
            ):
                raise ValueError("summary child has the wrong parent, kind or binding")
            ref = _ref(child, "validation/execution-plan.json")
            if ref is None or ref.run_id != child.id or ref.visibility != "public":
                raise ValueError("summary child has no public execution plan")
            plan = CaseExecutionPlan.model_validate_json(lease.read(ref))
            if (
                plan.case_id != case.case_id
                or plan.role != role_name
                or plan.split != "public"
                or plan.target_tool != case.target_tool
                or plan.sanitizer_repetitions != case.repetitions
                or plan.case_registry_hash != batch.binding.case_registry_hash
            ):
                raise ValueError("summary child does not match its case/role")
            runs.append(child)
    for run in runs:
        if any(ref.visibility != "public" or ref.run_id != run.id for ref in run.artifact_refs):
            raise ValueError("private or cross-run artifact cannot be exported")
    return runs


def export_batch(store: RunStore, batch_id: str, destination: Path) -> Path:
    """Export only summary-named public native runs through pinned RunStore leases."""
    if store.visibility != "public":
        raise ValueError("batch exports support the public store only")
    output = destination.absolute()
    reject_symlinks(output)
    output = output.resolve()
    if output.is_relative_to(store.root.resolve()):
        raise ValueError("export destination must be outside the RunStore")
    if output.exists():
        raise FileExistsError(output)
    with ExitStack() as stack:
        parent_lease = stack.enter_context(store.evaluation_run_lease(batch_id))
        batch = parent_lease.load()
        summary = _summary_from_manifest(batch, batch_id, parent_lease.read)
        leases = {batch_id: parent_lease}
        for case in summary.cases:
            for role in (case.clean, case.mutant):
                if role is None or role.run_id is None or role.run_id in leases:
                    continue
                leases[role.run_id] = stack.enter_context(store.evaluation_run_lease(role.run_id))
        runs = _selected_runs(batch, summary, leases)
        if sum(ref.byte_count for run in runs for ref in run.artifact_refs) > EXPORT_LIMIT:
            raise ValueError("batch export exceeds 256 MiB; export fewer cases")
        with output.open("xb") as stream:
            try:
                with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    inventory = []
                    names: set[str] = set()

                    def put(name: str, content: bytes) -> None:
                        path = PurePosixPath(name)
                        if (
                            name in names
                            or path.is_absolute()
                            or ".." in path.parts
                            or "\\" in name
                        ):
                            raise ValueError("unsafe or duplicate export member")
                        names.add(name)
                        archive.writestr(name, content)
                        inventory.append(
                            {
                                "path": name,
                                "sha256": hashlib.sha256(content).hexdigest(),
                                "byte_count": len(content),
                            }
                        )

                    put("summary.json", summary.model_dump_json(indent=2).encode())
                    put("report.md", render_batch(summary).encode())
                    put(
                        "EXPORT_NOTICE.txt",
                        b"Public seed diagnostic snapshot; not a restorable corpus, private "
                        b"verification audit, or signed evaluation result. Only summary-listed "
                        b"runs are exported. Inspect logs before sharing publicly.\n",
                    )
                    for run in runs:
                        lease = leases[run.id]
                        lease.validate()
                        put(f"runs/{run.id}/manifest.json", run.model_dump_json(indent=2).encode())
                        for ref in run.artifact_refs:
                            content = lease.read(ref)
                            put(f"runs/{run.id}/artifacts/{ref.id}/{ref.name}", content)
                    for lease in leases.values():
                        lease.validate()
                    archive.writestr("inventory.json", json.dumps(inventory, indent=2))
            except BaseException:
                output.unlink(missing_ok=True)
                raise
    return output
