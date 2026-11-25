#!/usr/bin/env python3
"""One paid CPU acceptance run: native pause/resume, exact edits, audit, and memory.

Uses only existing OPENAI_* / GPU_AGENT_* environment configuration. Never retries
a failed run; use a fresh output directory for every explicitly authorized run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit


class AcceptanceFailure(Exception):
    pass


def require(condition, code):
    if not condition:
        raise AcceptanceFailure(code)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def git(directory, *args):
    return subprocess.run(
        ["git", "-C", str(directory), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def persist(output, report):
    temporary = output / "acceptance-summary.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    temporary.replace(output / "acceptance-summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Implementation Git checkout")
    parser.add_argument("--output", type=Path, required=True, help="Fresh acceptance directory")
    parser.add_argument("--allow-paid-calls", action="store_true")
    args = parser.parse_args()
    if not args.allow_paid_calls:
        parser.error("--allow-paid-calls is required; no provider was constructed")
    output = args.output.absolute()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    started = time.monotonic()
    session, fixture, database = output / "session", output / "fixture", output / "memory.sqlite3"
    report = {
        "status": "RUNNING",
        "started_at": datetime.now(UTC).isoformat(),
        "phases": [],
        "model_calls": None,
        "model_tokens": None,
    }
    persist(output, report)
    try:
        source = args.source.resolve(strict=True)
        sys.path.insert(0, str(source / "src"))
        from gpu_agent.agent.provider import OpenAIProviderSettings, provider_invocations
        from gpu_agent.harness.graph_memory import memory_configuration
        from gpu_agent.harness.journal import read_verified_journal
        from gpu_agent.harness.native_graph import inspect_native_graph
        from gpu_agent.harness.repo_runner import run_repository_coding_session
        from gpu_agent.store import RunStore

        report.update(
            code_sha=git(source, "rev-parse", "HEAD"),
            source_diff_sha256=digest(git(source, "diff", "HEAD")),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        settings = OpenAIProviderSettings.from_environment()
        require(settings.api_key and settings.model and settings.endpoint, "MISSING_PROVIDER_ENV")
        require(urlsplit(settings.endpoint).scheme == "https", "REAL_HTTPS_PROVIDER_REQUIRED")
        report.update(model=settings.model, endpoint_host=urlsplit(settings.endpoint).hostname)
        fixture.mkdir(mode=0o700)
        original = {"kernel.cu": "float scale = 1.0f;\n", "bias.cu": "float bias = 0.0f;\n"}
        expected = {"kernel.cu": "float scale = 2.0f;\n", "bias.cu": "float bias = 1.0f;\n"}
        for name, contents in original.items():
            (fixture / name).write_text(contents, encoding="utf-8")
        git(fixture, "init", "-q")
        git(fixture, "add", "kernel.cu", "bias.cu")
        git(
            fixture,
            "-c",
            "user.name=Acceptance",
            "-c",
            "user.email=acceptance@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-qm",
            "CPU acceptance fixture",
        )
        options = dict(
            repository=fixture,
            output=session,
            base_sha=git(fixture, "rev-parse", "HEAD"),
            goal="Change kernel.cu from 'float scale = 1.0f;' to 'float scale = 2.0f;' and "
            "bias.cu from 'float bias = 0.0f;' to 'float bias = 1.0f;'. Preserve all other "
            "content, then finish. This CPU source fixture has no compiler, probe or test "
            "tools. Plan using inspect, edit and finish gates only.",
            max_steps=10,
            max_model_calls=18,
            max_edited_files=2,
            hierarchical_mode=True,
            runtime_mode="langgraph",
            provider_settings=settings,
            allow_paid_calls=True,
            memory_mode="learn",
            memory_database=database,
            memory_scope="live_" + uuid.uuid4().hex,
        )
        report.update(
            fixture_base_sha=options["base_sha"],
            goal_sha256=digest(options["goal"]),
            memory_scope=options["memory_scope"],
            max_steps=10,
            max_model_calls=18,
        )
        prior_effects, prior_invocations, prior_checkpoints = [], {}, 0
        for phase_name, phase_options in (
            ("pause", {"pause_after_turns": 1}),
            ("resume", {"resume": True}),
        ):
            phase_start = time.monotonic()
            phase = {
                "phase": phase_name,
                "status": "RUNNING",
                "started_at": datetime.now(UTC).isoformat(),
            }
            report["phases"].append(phase)
            persist(output, report)
            outcome = run_repository_coding_session(**options, **phase_options)
            phase.update(
                {
                    key: getattr(outcome, key)
                    for key in (
                        "status",
                        "reason",
                        "turns",
                        "tool_calls",
                        "model_calls",
                        "model_tokens",
                    )
                }
            )
            phase["elapsed_seconds"] = round(time.monotonic() - phase_start, 3)
            info = inspect_native_graph(session)
            effects = read_verified_journal(session / "langgraph-effects.jsonl")
            events = read_verified_journal(session / "session-events.jsonl")
            invocations = provider_invocations(RunStore(session / "public"), outcome.public_run_id)
            invocation_hashes = {
                item.invocation_id: digest(item.model_dump(mode="json")) for item in invocations
            }
            effect_hashes = [row["sha256"] for row in effects]
            admissions = [
                row["payload"]["effect_key"]
                for row in effects
                if row["kind"] in {"model_started", "tool_started"}
            ]
            phase.update(
                checkpoint_count=info["checkpoint_count"],
                checkpoint_id=info["checkpoint_id"],
                memory_status=info["memory_status"],
                graph_events_sha256=digest(info["graph_events"]),
                effects_head_sha256=effects[-1]["sha256"],
                session_events_head_sha256=events[-1]["sha256"],
                invocation_records_sha256=digest(invocation_hashes),
            )
            require(len(admissions) == len(set(admissions)), "DUPLICATE_EFFECT_ADMISSION")
            require(
                len(effects) == 1 + 2 * len(admissions)
                and info["effects_completed"] == len(admissions),
                "INCOMPLETE_EFFECT_JOURNAL",
            )
            require(
                sum(row["kind"] == "tool_started" for row in effects)
                == outcome.tool_calls
                == info["turns"]
                <= 10,
                "TOOL_AUDIT_ACCOUNTING",
            )
            require(
                outcome.public_checks_attempted == outcome.public_probes_attempted == 0,
                "UNEXPECTED_GPU_ACTION",
            )
            require(len(invocations) == outcome.model_calls <= 18, "PHYSICAL_CALL_ACCOUNTING")
            require(
                not any(item.state in {"STARTED", "UNCERTAIN"} for item in invocations),
                "UNCERTAIN_PROVIDER_CALL",
            )
            config, memory = memory_configuration(
                fixture, session, database, options["memory_scope"], "read", None
            )
            lesson = memory.get(tuple(config["namespace"]), outcome.public_run_id)
            if phase_name == "pause":
                require(
                    outcome.status == "PAUSED_RESUMABLE" and outcome.turns == 1,
                    "EXPECTED_ONE_TURN_PAUSE",
                )
                require(
                    info["reason"] == "SESSION_PAUSED" and lesson is None,
                    "PAUSE_CHECKPOINT_OR_MEMORY",
                )
                prior_effects, prior_invocations = effect_hashes, invocation_hashes
                prior_checkpoints = info["checkpoint_count"]
            else:
                require(
                    outcome.status == "DIFF_STAGED_UNVERIFIED"
                    and outcome.reason == "WORKSPACE_DIFF_STAGED_UNVERIFIED",
                    "EXPECTED_FINISH",
                )
                require(
                    effect_hashes[: len(prior_effects)] == prior_effects,
                    "ORIGINAL_EFFECT_JOURNAL_CHANGED",
                )
                require(
                    all(invocation_hashes.get(k) == v for k, v in prior_invocations.items()),
                    "ORIGINAL_PROVIDER_INVOCATION_CHANGED",
                )
                require(info["checkpoint_count"] > prior_checkpoints, "NO_RESUMED_CHECKPOINTS")
                require(
                    all(
                        (session / "worktree" / name).read_text() == contents
                        for name, contents in expected.items()
                    ),
                    "WRONG_EXACT_SOURCE_RESULT",
                )
                require(set(outcome.edited_files) == set(expected), "UNEXPECTED_EDITED_FILES")
                require(
                    info["memory_status"] == "LEARNED" and lesson is not None,
                    "MISSING_LEARNED_MEMORY",
                )
                require(
                    len(memory.search(tuple(config["namespace"]), limit=2)) == 1,
                    "UNEXPECTED_MEMORY_EPISODE_COUNT",
                )
                require(
                    lesson.value["run_id"] == outcome.public_run_id
                    and lesson.value["outcome"] == "STAGED_UNVERIFIED"
                    and lesson.value["effect_head_sha256"] == effects[-1]["sha256"],
                    "MEMORY_EPISODE_PROVENANCE",
                )
                report.update(
                    memory_episode_sha256=digest(lesson.value),
                    original_effects_preserved=True,
                    exact_files_verified=True,
                )
            phase["acceptance"] = "PASS"
            persist(output, report)
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error_type=type(exc).__name__)
        if report["phases"]:
            report["phases"][-1].update(
                acceptance="FAIL",
                error_type=type(exc).__name__,
                elapsed_seconds=round(time.monotonic() - phase_start, 3),
            )
        if isinstance(exc, AcceptanceFailure):
            report["failure_code"] = str(exc)
    finally:
        # Read persisted accounting even after a failure; never print exception messages,
        # provider settings, environment values, model text, or credentials.
        try:
            header = json.loads((session / "langgraph-runtime.json").read_text())
            records = provider_invocations(RunStore(session / "public"), header["run_id"])
            counts = [row.usage.total_tokens if row.usage else None for row in records]
            report.update(
                model_calls=len(records),
                model_tokens=sum(counts) if all(x is not None for x in counts) else None,
                provider_states=[row.state for row in records],
            )
        except Exception as exc:
            report["accounting_error_type"] = type(exc).__name__
        report["artifact_sha256"] = {
            name: hashlib.sha256((session / name).read_bytes()).hexdigest()
            for name in (
                "langgraph-effects.jsonl",
                "session-events.jsonl",
                "result.json",
                "session-checkpoint.json",
            )
            if (session / name).is_file()
        }
        report.update(
            finished_at=datetime.now(UTC).isoformat(),
            elapsed_seconds=round(time.monotonic() - started, 3),
        )
        persist(output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
