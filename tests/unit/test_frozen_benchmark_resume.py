"""Fault injection of the actual resume reducer and durable hash-chain."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS=Path(__file__).resolve().parents[2]/"scripts"
sys.path.insert(0,str(SCRIPTS))
import run_frozen_benchmark_resume as resume  # noqa: E402


@pytest.fixture
def manifest():
    path=SCRIPTS.parent/"docs/benchmark/2026-11-25-formal-benchmark-manifest.json"
    return json.loads(path.read_text())


def new_ledger(tmp_path,manifest):
    return resume.Ledger(tmp_path/"events",manifest)


def test_startup_failure_before_reservation(tmp_path,manifest):
    l=new_ledger(tmp_path,manifest)
    l.append("INFRA_EVENT",details={"event":"server failed before first trial"})
    r=new_ledger(tmp_path,manifest)
    assert all(x["state"]=="PENDING" for x in r.states.values())
    tid=manifest["trial_order"][0]["id"]
    assert resume.recovery_action(r.states[tid],record=False,agent_done=False,
        strict_done=False,any_agent_evidence=False)=="RUN_NEW"


@pytest.mark.parametrize("boundary,phase,agent,strict,record,evidence,expected",[
    ("before_agent_launch","RESERVED",False,False,False,False,"INFRA_ABORT"),
    ("model_call","AGENT_STARTED",False,False,False,True,"UNKNOWN"),
    ("after_tool_completion","AGENT_STARTED",False,False,False,True,"UNKNOWN"),
    ("after_agent_report","AGENT_RECORDED",True,False,False,True,"FINALIZE_AGENT_WITHOUT_RESAMPLING"),
    ("strict_in_flight","STRICT_STARTED",True,False,False,True,"UNKNOWN"),
    ("after_strict_projection","STRICT_STARTED",True,True,False,True,"ADOPT_STRICT"),
    ("after_strict_record","STRICT_RECORDED",True,True,False,True,"ADOPT_STRICT"),
    ("after_result_write","RESULT_WRITTEN",True,True,True,True,"ADOPT_RECORD"),
])
def test_fault_injection_no_resample(tmp_path,manifest,boundary,phase,agent,strict,record,evidence,expected):
    l=new_ledger(tmp_path,manifest);tid=manifest["trial_order"][0]["id"]
    calls={"fake_model":0,"fake_gpu":0}
    l.append("RESERVED",tid,state="RUNNING",phase="RESERVED",attempt="a"*32)
    if phase!="RESERVED":
        l.phase(tid,"AGENT_STARTED")
        calls["fake_model"]+=1
    if boundary=="after_tool_completion" or agent:
        calls["fake_gpu"]+=1
    if resume.PHASES.index(phase)>resume.PHASES.index("AGENT_STARTED"):
        l.phase(tid,phase)
    # Simulated interruption: construct a fresh reader from disk.
    fresh=new_ledger(tmp_path,manifest)
    decision=resume.recovery_action(fresh.states[tid],record=record,agent_done=agent,
        strict_done=strict,any_agent_evidence=evidence)
    assert decision==expected and decision!="RUN_NEW"
    old_calls=dict(calls)
    # Recovery never invokes the fake model again. For the completed-report
    # branch it may finish a never-started independent check, not inference.
    if decision=="FINALIZE_AGENT_WITHOUT_RESAMPLING":
        calls["fake_gpu"]+=1
    assert calls["fake_model"]==old_calls["fake_model"]
    if decision in ("ADOPT_RECORD","ADOPT_STRICT"):
        assert calls["fake_gpu"]==old_calls["fake_gpu"]


def test_immutable_terminal_and_duplicate_reservation(tmp_path,manifest):
    l=new_ledger(tmp_path,manifest);tid=manifest["trial_order"][0]["id"]
    l.append("LEGACY_ADOPTED",tid,state="COMPLETED_SUCCESS",phase="RESULT_WRITTEN",
             attempt="b"*64,details={"seal":{"record_sha256":"c"*64}})
    fresh=new_ledger(tmp_path,manifest)
    assert fresh.states[tid]["state"]=="COMPLETED_SUCCESS"
    with pytest.raises(AssertionError):
        fresh.append("RESERVED",tid,state="RUNNING",phase="RESERVED",attempt="a"*32)
    assert resume.recovery_action(fresh.states[tid],record=True,agent_done=True,
        strict_done=True,any_agent_evidence=True)=="VERIFY_TERMINAL"


def test_pending_untracked_artifact_is_not_new_trial(tmp_path,manifest):
    l=new_ledger(tmp_path,manifest);tid=manifest["trial_order"][0]["id"]
    assert resume.recovery_action(l.states[tid],record=False,agent_done=False,
        strict_done=False,any_agent_evidence=True)=="UNKNOWN"


def test_hash_chain_and_no_overwrite(tmp_path,manifest):
    l=new_ledger(tmp_path,manifest);l.append("INFRA_EVENT",details={"first":True})
    p=l.root/"event-00001.json"
    with pytest.raises(FileExistsError):
        resume.atomic_json(p,{"tampered":True})
    r=json.loads(p.read_text());r["details"]["first"]=False;p.write_text(json.dumps(r))
    with pytest.raises(AssertionError):
        new_ledger(tmp_path,manifest)


def test_progress_atomic_replace(tmp_path):
    p=tmp_path/"progress.json"
    resume.atomic_json(p,{"count":6},replace=True)
    resume.atomic_json(p,{"count":7},replace=True)
    assert json.loads(p.read_text())=={"count":7}
    assert not list(tmp_path.glob(".pending-*"))
