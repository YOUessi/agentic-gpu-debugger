"""Resume the precommitted P8 trial order without resampling any attempt.

Infrastructure only. The original benchmark driver and the actual Harness
remain frozen. Every reservation and phase is append-only and hash-chained.
Existing evidence is verified before adoption, never rewritten.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import run_frozen_benchmark_repetitions as frozen

INFRA = "p8-infra-v2"
MANIFEST_SHA = "47d38ec2b9c363dd4acb29074f46aad6dd9913dac2bc35bc10bcd0d3ba050610"
ORIGINAL_RUNNER_SHA = "83c9596269b14bae146adf948fde86916d73c0d208c9798288990763b80e0f7d"
TERMINAL = {"COMPLETED_SUCCESS", "COMPLETED_MODEL_FAILURE", "INFRA_ABORTED", "OUTCOME_UNKNOWN"}
PHASES = ("RESERVED", "AGENT_STARTED", "AGENT_RECORDED", "AUDIT_RECORDED",
          "STRICT_STARTED", "STRICT_RECORDED", "RESULT_WRITTEN")
LEGACY_IDS = {
    "r01-deepseek-F-h001", "r01-deepseek-F-h002", "r01-deepseek-F-h003",
    "r01-deepseek-R-A-h001", "r01-deepseek-R-B-h001", "r01-deepseek-R-A-h003",
}
MODEL_FAILURES = {
    "STEP_BUDGET_EXHAUSTED", "LLM_REPOSITORY_ROLE_BUDGET_EXHAUSTED",
    "LLM_INVALID_OUTPUT", "LLM_INCOMPLETE_MAX_OUTPUT_TOKENS",
    "LLM_INCOMPLETE_CONTENT_FILTER", "LLM_INCOMPLETE_OTHER", "LLM_REFUSED",
    "LLM_BUDGET_EXCEEDED", "LLM_CALL_BUDGET_EXHAUSTED", "LLM_WALL_TIME_BUDGET_EXHAUSTED",
}
INFRA_ERRORS = {
    "LLM_AUTHENTICATION_FAILED", "LLM_PERMISSION_DENIED", "LLM_RATE_LIMITED",
    "LLM_MODEL_OR_ENDPOINT_NOT_FOUND", "LLM_INVALID_REQUEST", "LLM_PROVIDER_ERROR",
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def h(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required immutable record unavailable: {path.name}")
    return json.loads(path.read_bytes())


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    """fsync contents; publish atomically; never overwrite immutable records."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(tmp, path)
        else:
            # An atomic no-overwrite publication, unlike rename over a record.
            os.link(tmp, path)
            tmp.unlink()
        fsync_dir(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink()


def trial_identity(manifest: dict, trial: dict, unit: dict) -> dict:
    model_config = {"model_system": manifest["models"][trial["model"]],
                    "server_seed": trial["server_seed"]}
    identity = {
        "benchmark_manifest_sha256": MANIFEST_SHA,
        "model_system_config_sha256": h(model_config),
        "task_id": unit["case_id"],
        "evaluation_unit": unit["id"],
        "start_mode": "from_scratch" if unit["failure_kind"] == "none" else "seeded_recovery",
        "start_candidate_sha256": unit.get("prior_candidate_sha256"),
        "replicate_index": trial["rep"],
    }
    return {**identity, "trial_key": h(identity), "trial_id": trial["id"]}


class Ledger:
    def __init__(self, root: Path, manifest: dict):
        self.root = root
        self.units = {u["id"]:u for u in manifest["units"]}
        self.trials = {t["id"]:t for t in manifest["trial_order"]}
        self.identities = {tid:trial_identity(manifest,t,self.units[t["unit"]])
                           for tid,t in self.trials.items()}
        self.states = {tid:{"state":"PENDING","phase":None,"attempt_id":None}
                       for tid in self.trials}
        self.records: list[dict] = []
        self.head = "0"*64
        if root.exists():
            for p in sorted(root.glob("event-*.json")):
                record = read(p)
                if p.name != f"event-{len(self.records)+1:05d}.json":
                    raise ValueError("ledger sequence has a gap")
                self._accept(record)
        assert len(set(i["trial_key"] for i in self.identities.values())) == len(self.trials)

    def _accept(self, event: dict) -> None:
        value = dict(event)
        event_hash = value.pop("event_sha256")
        assert event_hash == h(value) and value["previous_sha256"] == self.head
        assert value["sequence"] == len(self.records)+1
        assert value["manifest_sha256"] == MANIFEST_SHA
        tid, kind = value["trial_id"], value["kind"]
        if tid is None:
            assert kind in {"INFRA_EVENT","SCHEDULER_STARTED","SCHEDULER_STOPPED","HEALTH_PASSED"}
        else:
            assert value["identity"] == self.identities[tid]
            old = self.states[tid]
            state, phase, attempt = value["state"],value["phase"],value["attempt_id"]
            if kind == "LEGACY_ADOPTED":
                assert old["state"] == "PENDING" and state in TERMINAL
                assert value["details"].get("seal")
            elif kind == "RESERVED":
                assert old["state"] == "PENDING" and state == "RUNNING" and phase == "RESERVED"
            elif kind == "PHASE":
                assert old["state"] == state == "RUNNING" and old["attempt_id"] == attempt
                assert PHASES.index(phase) > PHASES.index(old["phase"])
            elif kind == "TERMINAL":
                assert old["state"] == "RUNNING" and state in TERMINAL
                assert old["attempt_id"] == attempt
                if state.startswith("COMPLETED_"):
                    assert value["details"].get("seal")
            else:
                raise ValueError("unknown trial ledger event")
            assert re.fullmatch(r"[0-9a-f]{32,64}",attempt)
            self.states[tid] = {"state":state,"phase":phase,"attempt_id":attempt,
                                "details":value["details"]}
        self.records.append(event)
        self.head = event_hash

    def append(self, kind, tid=None, *, state=None, phase=None, attempt=None, details=None):
        body = {"sequence":len(self.records)+1,"previous_sha256":self.head,
                "manifest_sha256":MANIFEST_SHA,"infra_version":INFRA,
                "kind":kind,"trial_id":tid,
                "identity":self.identities[tid] if tid is not None else None,
                "state":state,"phase":phase,"attempt_id":attempt,
                "details":details or {},"unix_seconds":time.time()}
        record = {**body,"event_sha256":h(body)}
        # Validate before publication; roll back in-memory state if disk fails.
        old = self.states[tid] if tid is not None else None
        previous = self.head
        self._accept(record)
        try:
            atomic_json(self.root/f"event-{body['sequence']:05d}.json",record)
        except BaseException:
            self.records.pop()
            self.head = previous
            if tid is not None:
                self.states[tid] = old
            raise
        return record

    def phase(self, tid, phase, details=None):
        return self.append("PHASE", tid, state="RUNNING", phase=phase,
                           attempt=self.states[tid]["attempt_id"], details=details)

    def terminal(self, tid, state, details):
        return self.append("TERMINAL",tid,state=state,phase=self.states[tid]["phase"],
                           attempt=self.states[tid]["attempt_id"],details=details)


def recovery_action(state: dict, *, record: bool, agent_done: bool,
                    strict_done: bool, any_agent_evidence: bool) -> str:
    """Never return RUN_NEW for a previously reserved trial."""
    if state["state"] in TERMINAL:
        return "VERIFY_TERMINAL" if state["state"].startswith("COMPLETED_") else "HALT"
    if state["state"] == "PENDING":
        return "UNKNOWN" if any_agent_evidence or record else "RUN_NEW"
    assert state["state"] == "RUNNING"
    if record:
        return "ADOPT_RECORD"
    if state["phase"] == "RESERVED" and not any_agent_evidence:
        return "INFRA_ABORT"
    if state["phase"] in ("STRICT_STARTED","STRICT_RECORDED","RESULT_WRITTEN"):
        return "ADOPT_STRICT" if agent_done and strict_done else "UNKNOWN"
    if agent_done:
        return "FINALIZE_AGENT_WITHOUT_RESAMPLING"
    return "UNKNOWN"


def public_inventory(store, root_id: str) -> dict:
    """Verify every registered public artifact, but never open the private store."""
    queue = [root_id]
    runs = []
    while queue:
        rid = queue.pop(0)
        m = store.load(rid)
        refs = []
        for ref in m.artifact_refs:
            raw = store.read(ref)
            refs.append({"id":ref.id,"name":ref.name,"sha256":frozen.digest(raw),
                         "bytes":len(raw)})
        manifest = store.root / rid / "manifest.json"
        runs.append({"id":rid,"kind":m.kind,"manifest_sha256":frozen.file_digest(manifest),
                     "refs":refs})
        queue.extend(sorted(child.id for child in store.children(rid)))
    return {"run_count":len(runs),"artifact_count":sum(len(x["refs"]) for x in runs),
            "inventory_sha256":h(runs)}


def pending_effects(journal):
    pending = set()
    for entry in journal:
        phase, turn = entry["kind"],entry["payload"].get("turn")
        if phase=="model_started":
            pending.add(("model",turn))
        elif phase=="model_selected":
            pending.discard(("model",turn))
        elif phase=="tool_started":
            pending.add(("tool",turn))
        elif phase=="tool_completed":
            pending.discard(("tool",turn))
    return pending


def verify_strict_projection(folder: Path, audit: dict, unit: dict) -> dict:
    from gpu_agent.benchmark.hardset_v2_verifier import HardsetPrivateProjection
    from gpu_agent.store import RunStore

    p = folder/"strict-projection.json"
    projection = HardsetPrivateProjection.model_validate_json(p.read_bytes()).model_dump(mode="json")
    assert projection["candidate_sha256"] == audit["public_passed_candidate_sha256"]
    assert projection["candidate_sha256"] == audit["final_candidate_sha256"]
    assert projection["case_id"] == unit["case_id"]
    root = folder/"strict-verification/public"
    store = RunStore(root)
    runs = [store.load(x.name) for x in root.iterdir()
            if x.is_dir() and re.fullmatch(r"[0-9a-f]{32}",x.name)]
    assert len(runs)==1 and runs[0].kind=="hardset_v2_strict_projection"
    m = runs[0]
    refs = [r for r in m.artifact_refs if r.name=="hardset/strict-verdict.json"]
    assert len(refs)==1 and json.loads(store.read(refs[0]))==projection
    assert m.status.value in ("COMPLETED","FAILED")
    return {"status":projection["verdict"],"projection":projection,
            "strict_public_run_id":m.id,"projection_sha256":frozen.file_digest(p),
            "public_inventory":public_inventory(store,m.id)}


def verify_agent(repo: Path, folder: Path, unit: dict, trial: dict) -> tuple[dict,dict,str]:
    from gpu_agent.agent.provider import provider_invocations
    from gpu_agent.harness.journal import read_verified_journal
    from gpu_agent.harness.models import RepoHarnessOutcome
    from gpu_agent.harness.recovery import read_repository_checkpoint
    from gpu_agent.harness.repo_audit import read_repository_tool_events
    from gpu_agent.store import RunStore

    session = folder/"session"
    model_id = "deepseek-v4-pro" if trial["model"]=="deepseek" else "qwen3-8b-awq"
    audit = frozen.audit_session(repo,session,unit,model_id)
    agent = session/"agent"
    journal = read_verified_journal(agent/"session-events.jsonl")
    store = RunStore(agent/"public")
    rid = audit["public_run_id"]
    inv = provider_invocations(store,rid)
    parent = store.load(rid)
    events = read_repository_tool_events(store,rid,expected_count=audit["total_turns"],
                                         require_action=True,require_observation=True)
    n = len(events)
    checkpoint_path = agent/"session-checkpoint.json"
    checkpoint_hash = None
    if n:
        cp = read_repository_checkpoint(checkpoint_path)
        assert cp.public_run_id==rid and cp.base_sha==frozen.FROZEN
        assert cp.state["turns"]==n and cp.max_steps==16 and cp.max_model_calls==18
        assert cp.max_output_tokens==8192 and cp.max_compiles==6 and cp.max_public_checks==2
        assert cp.ast_target_binding and cp.state_scoped_actions
        assert cp.public_case_id==unit["case_id"]
        assert cp.compile_gate==unit["compile_gate"] and cp.allow_revert==unit["allow_revert"]
        assert cp.public_diagnostics==unit["public_diagnostics"]
        assert cp.diff_sha256==events[-1]["diff_sha256"]
        completed = [x for x in journal if x["kind"]=="tool_completed"]
        assert len(completed)==n and cp.journal_head_sha256==completed[-1]["sha256"]
        for entry, event in zip(cp.state["history"],events,strict=True):
            assert entry=={"tool":event["tool"],"status":event["status"],
                           "content":event["observation"][:3900]}
        assert cp.state["decision"]["action"]==events[-1]["action"]
        checkpoint_hash=frozen.file_digest(checkpoint_path)
    else:
        assert not checkpoint_path.exists()
    result_path=agent/"result.json"
    result_hash=None
    reason=audit["terminal_reason"]
    terminal_kind=journal[-1]["kind"]
    if result_path.exists():
        outcome=RepoHarnessOutcome.model_validate_json(result_path.read_bytes())
        assert outcome.base_sha==frozen.FROZEN and outcome.public_run_id==rid
        assert outcome.tool_calls==n and outcome.model_calls==len(inv)
        assert outcome.reason==reason
        assert terminal_kind=="session_finished" and journal[-1]["payload"]["status"]==outcome.status
        assert not pending_effects(journal)
        if n:
            assert outcome.diff_sha256==events[-1]["diff_sha256"]
        assert outcome.public_checks_passed==bool(audit["public_passed_candidate_sha256"])
        result_hash=frozen.file_digest(result_path)
    else:
        assert terminal_kind=="session_failed"
        assert not any(x[0]=="tool" for x in pending_effects(journal))
    assert parent.status.value in ("COMPLETED","FAILED")
    if unit["failure_kind"]!="none" and not audit["initial_failure_reproduced"]:
        category="INFRA_ABORTED"
    elif any(x.state in ("STARTED","UNCERTAIN") for x in inv):
        category="OUTCOME_UNKNOWN"
    elif reason in INFRA_ERRORS:
        category="INFRA_ABORTED"
    elif reason in MODEL_FAILURES:
        # Known deterministic terminal provider/budget error is not an
        # uncertain physical side effect merely because model_started remains.
        if reason=="LLM_REPOSITORY_ROLE_BUDGET_EXHAUSTED":
            assert sum(x.attempt==1 for x in inv)==2 or len(inv)>=16
        category="COMPLETED_MODEL_FAILURE"
    elif audit["public_passed_candidate_sha256"]:
        category="PUBLIC_PASS"
    else:
        category="OUTCOME_UNKNOWN"
    if audit["public_passed_candidate_sha256"] and category!="OUTCOME_UNKNOWN":
        category="PUBLIC_PASS"
    if category=="COMPLETED_MODEL_FAILURE":
        if any(x["status"]=="BUILD_UNAVAILABLE" for x in audit["build_results"]):
            category="INFRA_ABORTED"
        if any(x["failure_code"]=="PUBLIC_EXECUTION_UNAVAILABLE" for x in audit["live_checks"]):
            category="INFRA_ABORTED"
    seal={"journal_sha256":frozen.file_digest(agent/"session-events.jsonl"),
          "journal_head_sha256":journal[-1]["sha256"],
          "checkpoint_sha256":checkpoint_hash,"result_sha256":result_hash,
          "controlled_recovery_sha256":frozen.file_digest(session/"controlled-recovery.json"),
          "experiment_config_sha256":frozen.file_digest(session/"experiment-config.json"),
          "final_source_sha256":audit["final_candidate_sha256"],
          "public_inventory":public_inventory(store,rid),
          "invocations_sha256":h([x.model_dump(mode="json") for x in inv]),
          "invocation_count":len(inv),"terminal_reason":reason}
    return audit,seal,category


def verify_record(a, trial, unit, folder):
    reservation=read(folder/"attempt-reserved.json")
    assert reservation["trial"]==trial and reservation["unit"]==unit
    assert reservation["manifest_sha256"]==MANIFEST_SHA
    record=read(folder/"trial-record.json")
    assert record["trial"]==trial and record["unit"]==unit
    assert type(record["success"]) is bool and isinstance(record["agent_execution"],dict)
    assert not record["agent_execution"]["timed_out"]
    audit,seal,category=verify_agent(a.repository,folder,unit,trial)
    assert audit==record["audit"]
    strict=record["strict"]
    if strict["status"]=="VERIFIED_FIXED" or strict["status"]=="NOT_VERIFIED":
        verified=verify_strict_projection(folder,audit,unit)
        for key in ("status","projection","strict_public_run_id","projection_sha256"):
            assert strict[key]==verified[key]
        seal["strict_public_inventory"]=verified["public_inventory"]
        seal["strict_projection_sha256"]=verified["projection_sha256"]
        category=("COMPLETED_SUCCESS" if verified["status"]=="VERIFIED_FIXED"
                  else "COMPLETED_MODEL_FAILURE")
        assert record["success"]==(category=="COMPLETED_SUCCESS")
    elif strict["status"]=="NOT_PUBLIC_PASSED":
        assert not audit["public_passed_candidate_sha256"] and not record["success"]
    else:
        raise ValueError("unresolved strict outcome is not an adoptable completed record")
    assert category in TERMINAL and category!="OUTCOME_UNKNOWN"
    if category.startswith("COMPLETED_"):
        assert record["status"]==("SUCCESS" if record["success"] else "FAILURE")
    return record,{**seal,"record_sha256":frozen.file_digest(folder/"trial-record.json"),
                    "reservation_sha256":frozen.file_digest(folder/"attempt-reserved.json")},category


def progress(a, ledger):
    states=collections.Counter(x["state"] for x in ledger.states.values())
    cells=[]
    for model in ("deepseek","qwen"):
        for uid,unit in ledger.units.items():
            labels=[k for k,t in ledger.trials.items() if t["model"]==model and t["unit"]==uid]
            count=collections.Counter(ledger.states[k]["state"] for k in labels)
            n=count["COMPLETED_SUCCESS"]+count["COMPLETED_MODEL_FAILURE"]
            cells.append({"model":model,"unit":uid,"group":unit["group"],"planned":len(labels),
                          "states":dict(count),"valid_model_trials":n,
                          "strict_successes":count["COMPLETED_SUCCESS"],
                          "wilson_95":frozen.wilson(count["COMPLETED_SUCCESS"],n) if n==5 else None})
    summary={"infra_version":INFRA,"manifest_sha256":MANIFEST_SHA,"ledger_head":ledger.head,
             "total":len(ledger.trials),"states":dict(states),"cells":cells,
             "completed_model_trials":states["COMPLETED_SUCCESS"]+states["COMPLETED_MODEL_FAILURE"],
             "trial_states":ledger.states}
    atomic_json(a.output/"scheduler-v2/progress.json",summary,replace=True)
    return summary


def adopt_existing(a, ledger, *, write):
    if frozen.file_digest(a.output/"frozen-manifest.json")!=MANIFEST_SHA:
        # The original runner serialized the same object, rather than copying
        # its exact bytes. Verify semantic equality and original preflight hash.
        assert read(a.output/"frozen-manifest.json")==read(a.manifest)
    original=read(a.output/"preflight.json")
    assert original["manifest_sha256"]==MANIFEST_SHA
    assert original["runner_sha256"]==ORIGINAL_RUNNER_SHA
    expected_dirs={p.name for p in (a.output/"trials").iterdir() if p.is_dir()}
    assert expected_dirs<=set(ledger.trials)
    found={}
    for tid,t in ledger.trials.items():
        state=ledger.states[tid]
        folder=a.output/"trials"/tid
        if state["state"].startswith("COMPLETED_") or (tid in LEGACY_IDS and state["state"]=="PENDING"):
            record,seal,category=verify_record(a,t,ledger.units[t["unit"]],folder)
            if state["state"]!="PENDING":
                assert state["state"]==category and state["details"]["seal"]==seal
            elif write:
                attempt=h({"legacy_reservation_sha256":seal["reservation_sha256"]})
                ledger.append("LEGACY_ADOPTED",tid,state=category,phase="RESULT_WRITTEN",
                              attempt=attempt,details={"seal":seal,"legacy":True})
            found[tid]={"state":category,"seal":seal,"strict":record["strict"]["status"]}
        elif state["state"]=="PENDING" and folder.exists():
            raise ValueError("untracked existing trial cannot be sampled")
    assert LEGACY_IDS<=set(found)
    return found


def isolated_env(a):
    env=dict(os.environ)
    for k in ("PYTHONPATH","PYTHONHOME","VIRTUAL_ENV","OPENAI_API_KEY","DEEPSEEK_API_KEY",
              "OPENAI_LOG","CONDA_PREFIX","CONDA_DEFAULT_ENV"):
        env.pop(k,None)
    env.update(PYTHONNOUSERSITE="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
               VLLM_LOGGING_LEVEL="WARNING",PATH=str(a.serving_python.parent)+":/usr/bin:/bin")
    return env


def serve_argv(a, seed):
    return [str(a.serving_python),"-I","-m","vllm.entrypoints.cli.main","serve",str(a.qwen_model),
            "--served-model-name","qwen3-8b-awq","--host","127.0.0.1","--port","8127",
            "--max-model-len","16384","--gpu-memory-utilization","0.65","--enforce-eager",
            "--reasoning-parser","qwen3","--seed",str(seed)]


def environment_preflight(a,m):
    assert frozen.file_digest(a.manifest)==MANIFEST_SHA
    assert frozen.file_digest(Path(frozen.__file__))==ORIGINAL_RUNNER_SHA
    # frozen.preflight checks the original weights, CUDA image, runtime Git
    # objects and controller dependency versions. Point its version check at
    # the already isolated absolute interpreter, not a PATH-resolved launcher.
    a.vllm=a.serving_python
    p=frozen.preflight(a,m)
    report=read(a.environment_report)
    code=r"""
import sys, pathlib, importlib.metadata as m, json, torch, vllm, typing_extensions
p=pathlib.Path(sys.prefix).resolve()
assert sys.flags.isolated==1 and sys.flags.no_user_site==1
assert pathlib.Path(sys.executable).parent.parent.resolve()==p
assert all(pathlib.Path(x.__file__).resolve().is_relative_to(p)
           for x in (torch,vllm,typing_extensions))
assert torch.__version__.split('+')[0]=='2.10.0' and torch.version.cuda=='12.8'
assert vllm.__version__=='0.19.1' and m.version('typing_extensions')=='4.15.0'
assert not any('/.local/' in x or ('site-packages' in x and not pathlib.Path(x).resolve().is_relative_to(p))
               or 'dist-packages' in x for x in sys.path)
print(json.dumps({'executable':sys.executable,'prefix':str(p),
                  'torch':{'path':torch.__file__,'version':torch.__version__,'cuda':torch.version.cuda},
                  'vllm':{'path':vllm.__file__,'version':vllm.__version__},
                  'typing_extensions':{'path':typing_extensions.__file__,'version':m.version('typing_extensions')},
                  'sys_path':sys.path}))
"""
    env=isolated_env(a)
    raw=subprocess.check_output([str(a.serving_python),"-I","-c",code],env=env,text=True,timeout=60)
    identity=json.loads(raw.splitlines()[-1])
    assert identity["executable"]==str(a.serving_python)
    assert report["target_python"]==str(a.serving_python)
    freeze=subprocess.check_output([str(a.serving_python),"-I","-m","pip","freeze","--all"],
                                   env=env,text=True,timeout=45)
    assert frozen.digest(freeze.encode())==report["pip_freeze_sha256"]
    subprocess.run([str(a.serving_python),"-I","-m","pip","check"],env=env,check=True,timeout=60)
    return {"infra_version":INFRA,"preflight":p,"module_identity":identity,
            "environment_report_sha256":frozen.file_digest(a.environment_report),
            "pip_freeze":freeze,"pip_freeze_sha256":report["pip_freeze_sha256"],
            "serving_commands":{str(seed):{"argv":serve_argv(a,seed),
                             "sha256":h(serve_argv(a,seed))} for seed in range(43,48)},
            "isolated_flags":{"unset":["PYTHONPATH","PYTHONHOME"],"PYTHONNOUSERSITE":"1",
                              "python_isolated_flag":"-I"}}


def local_health(a,seed):
    from typing import Literal
    import openai
    from pydantic import BaseModel, ConfigDict
    from openai.lib._pydantic import to_strict_json_schema

    class Health(BaseModel):
        model_config=ConfigDict(extra="forbid",strict=True)
        ok: Literal[True]

    with openai.OpenAI(base_url="http://127.0.0.1:8127/v1",api_key="local-health-check",
                       timeout=90,max_retries=0) as client:
        raw=client.responses.with_raw_response.parse(
            model="qwen3-8b-awq",instructions="Return only the requested JSON object. No explanation.",
            input="Infrastructure health check: set ok to true.",text_format=Health,
            max_output_tokens=8192,store=False)
        response=raw.parse()
        envelope=json.loads(raw.content)
        assert response.status=="completed" and response.model=="qwen3-8b-awq"
        assert response.output_parsed is not None and response.output_parsed.ok is True
        return {"non_benchmark":True,"physical_calls":1,"model":response.model,
                "status":response.status,"response_id":response.id,
                "schema_sha256":h(to_strict_json_schema(Health)),
                "usage":envelope.get("usage"),"seed":seed,"command_sha256":h(serve_argv(a,seed))}


@contextlib.contextmanager
def server(a,seed,log):
    assert frozen.gpu_state()["memory_used_mib"]<1000
    argv=serve_argv(a,seed)
    # Refuse an existing server before starting an owned process.
    import socket
    with socket.socket() as sock:
        sock.settimeout(0.2)
        assert sock.connect_ex(("127.0.0.1",8127))!=0
    with log.open("xb") as stream:
        proc=subprocess.Popen(argv,env=isolated_env(a),stdout=stream,stderr=subprocess.STDOUT,
                              start_new_session=True)
        try:
            import urllib.request
            deadline=time.monotonic()+180
            while time.monotonic()<deadline:
                if proc.poll() is not None:
                    raise ValueError("isolated local server exited")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8127/v1/models",timeout=2) as resp:
                        names=[x["id"] for x in json.load(resp)["data"]]
                    if names==["qwen3-8b-awq"]:
                        break
                except (OSError,ValueError,KeyError):
                    pass
                time.sleep(2)
            else:
                raise ValueError("isolated service readiness timed out")
            health=local_health(a,seed)
            print("NON_BENCHMARK_HEALTH",json.dumps(health),flush=True)
            yield health
        finally:
            frozen.stop_owned(proc)


def trial_argv(a,folder,unit):
    argv=[sys.executable,str(a.repository/"scripts/run_repository_failed_candidate_recovery.py"),
          "--repository",str(a.repository),"--output",str(folder/"session"),
          "--case",unit["case_id"],"--failure-kind",unit["failure_kind"],
          "--max-steps","16","--max-model-calls","18","--max-output-tokens","8192",
          "--max-compiles","6","--ast-target-binding","--state-scoped-actions",
          "--allow-paid-calls","--allow-real-gpu"]
    for flag,enabled in (("compile-gate",unit["compile_gate"]),("allow-revert",unit["allow_revert"]),
                         ("public-diagnostics",unit["public_diagnostics"])):
        if enabled:
            argv.append("--"+flag)
    if unit["failure_kind"]!="none":
        prior=a.evidence_root/unit["prior_path"]
        assert frozen.file_digest(prior)==unit["prior_sha256"]
        argv+=["--prior-result",str(prior),"--prior-result-sha256",unit["prior_sha256"]]
    return argv


def terminal_record(a,ledger,tid,record):
    folder=a.output/"trials"/tid
    atomic_json(folder/"trial-record.json",record)
    ledger.phase(tid,"RESULT_WRITTEN",{"record_sha256":frozen.file_digest(folder/"trial-record.json")})
    trial=ledger.trials[tid];unit=ledger.units[trial["unit"]]
    _,seal,category=verify_record(a,trial,unit,folder)
    ledger.terminal(tid,category,{"seal":seal})
    return category


def finish_agent(a,ledger,tid,env,*,existing_strict=False):
    trial=ledger.trials[tid];unit=ledger.units[trial["unit"]]
    folder=a.output/"trials"/tid
    if (folder/"agent-execution.json").is_file():
        execution=read(folder/"agent-execution.json")
        assert not execution["timed_out"]
    else:
        # The frozen CLI completed and persisted its public report, but the
        # outer process timing was not durably written. Do not invent timing.
        execution={"exit_code":None,"timed_out":False,"wall_seconds":None}
        atomic_json(folder/"agent-execution.json",execution)
    audit,agent_seal,category=verify_agent(a.repository,folder,unit,trial)
    if (folder/"public-audit.json").exists():
        assert read(folder/"public-audit.json")=={"audit":audit,"seal":agent_seal}
    else:
        atomic_json(folder/"public-audit.json",{"audit":audit,"seal":agent_seal})
    if PHASES.index(ledger.states[tid]["phase"])<PHASES.index("AUDIT_RECORDED"):
        ledger.phase(tid,"AUDIT_RECORDED",{"audit_sha256":frozen.file_digest(folder/"public-audit.json")})
    if category in ("OUTCOME_UNKNOWN","INFRA_ABORTED") and not audit["public_passed_candidate_sha256"]:
        ledger.terminal(tid,category,{"agent_seal":agent_seal,"no_resample":True})
        return category
    if (folder/"strict-execution.json").exists():
        strict=read(folder/"strict-execution.json")
    elif existing_strict:
        # Strict Fix has already completed; independently validate its exact
        # public projection. No second hidden check and no new model call.
        verified=verify_strict_projection(folder,audit,unit)
        strict={k:v for k,v in verified.items() if k!="public_inventory"}
        strict.update({"mode":"adopted-complete-projection","exit_code":None,
                       "timed_out":False,"wall_seconds":None})
        atomic_json(folder/"strict-execution.json",strict)
    elif audit["public_passed_candidate_sha256"]:
        ledger.phase(tid,"STRICT_STARTED",{"candidate_sha256":audit["public_passed_candidate_sha256"]})
        strict=frozen.strict_check(a,folder,unit,audit,env)
        atomic_json(folder/"strict-execution.json",strict)
    else:
        strict={"status":"NOT_PUBLIC_PASSED","projection":None}
        atomic_json(folder/"strict-execution.json",strict)
    if PHASES.index(ledger.states[tid]["phase"])<PHASES.index("STRICT_RECORDED"):
        ledger.phase(tid,"STRICT_RECORDED",{"strict_record_sha256":frozen.file_digest(folder/"strict-execution.json")})
    if strict["status"] not in ("VERIFIED_FIXED","NOT_VERIFIED","NOT_PUBLIC_PASSED"):
        ledger.terminal(tid,"OUTCOME_UNKNOWN",{"agent_seal":agent_seal,"strict":strict,"no_resample":True})
        return "OUTCOME_UNKNOWN"
    success=strict["status"]=="VERIFIED_FIXED"
    correct=None
    if success:
        sha=strict["projection"]["candidate_sha256"]
        correct=min(x["turn"] for x in audit["live_edits"] if x["sha256"]==sha)
    record={"trial":trial,"unit":unit,"agent_execution":execution,"audit":audit,"strict":strict,
            "first_correct_edit_turn":correct,"success":success,
            "status":"SUCCESS" if success else "FAILURE","halt_suite":False}
    return terminal_record(a,ledger,tid,record)


def run_new(a,ledger,tid,env):
    trial=ledger.trials[tid];unit=ledger.units[trial["unit"]]
    folder=a.output/"trials"/tid
    assert not folder.exists() and ledger.states[tid]["state"]=="PENDING"
    attempt=uuid.uuid4().hex
    argv=trial_argv(a,folder,unit)
    ledger.append("RESERVED",tid,state="RUNNING",phase="RESERVED",attempt=attempt,
                  details={"argv_sha256":h(argv)})
    folder.mkdir(mode=0o700)
    atomic_json(folder/"attempt-reserved.json",{"trial":trial,"unit":unit,
                "manifest_sha256":MANIFEST_SHA,"reserved_unix_seconds":time.time(),
                "trial_identity":ledger.identities[tid],"attempt_id":attempt})
    ledger.phase(tid,"AGENT_STARTED",{"argv_sha256":h(argv)})
    print("TRIAL_START",tid,attempt,flush=True)
    result=frozen.invoke(argv,env,folder/"agent-stdout.log",folder/"agent-stderr.log",600,a.repository)
    atomic_json(folder/"agent-execution.json",result)
    ledger.phase(tid,"AGENT_RECORDED",{"execution_sha256":frozen.file_digest(folder/"agent-execution.json")})
    if result["timed_out"]:
        ledger.terminal(tid,"OUTCOME_UNKNOWN",{"watchdog":True,"no_resample":True})
        return "OUTCOME_UNKNOWN"
    return finish_agent(a,ledger,tid,env)


def recover_running(a,ledger,tid):
    folder=a.output/"trials"/tid
    record=(folder/"trial-record.json").is_file()
    agent_done=(folder/"session/controlled-recovery.json").is_file()
    strict_done=(folder/"strict-execution.json").is_file() or (folder/"strict-projection.json").is_file()
    evidence=(folder/"session").exists() or (folder/"agent-stdout.log").exists()
    action=recovery_action(ledger.states[tid],record=record,agent_done=agent_done,
                           strict_done=strict_done,any_agent_evidence=evidence)
    if action=="ADOPT_RECORD":
        trial=ledger.trials[tid];unit=ledger.units[trial["unit"]]
        _,seal,category=verify_record(a,trial,unit,folder)
        ledger.terminal(tid,category,{"seal":seal,"recovered_after_record_write":True})
        return category
    if action in ("FINALIZE_AGENT_WITHOUT_RESAMPLING","ADOPT_STRICT"):
        env=frozen.session_environment(a,ledger.trials[tid]["model"])
        return finish_agent(a,ledger,tid,env,existing_strict=action=="ADOPT_STRICT")
    category="INFRA_ABORTED" if action=="INFRA_ABORT" else "OUTCOME_UNKNOWN"
    ledger.terminal(tid,category,{"recovery_action":action,"no_resample":True,
        "reason":"partial frozen recovery CLI has no audited same-budget resume; preserve attempt"})
    return category


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","repository","evidence-root","output","provider-config",
                 "evaluator-secret-file","qwen-model","serving-python","gpu-lock",
                 "environment-report","infra-output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--health-check",action="store_true")
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--allow-paid-calls",action="store_true")
    parser.add_argument("--allow-real-gpu",action="store_true")
    a=parser.parse_args()
    os.umask(0o077)
    for key in ("manifest","repository","evidence_root","output","provider_config",
                "evaluator_secret_file","qwen_model","serving_python","gpu_lock","environment_report"):
        setattr(a,key,getattr(a,key).expanduser().resolve(strict=True))
    a.infra_output=a.infra_output.expanduser().absolute()
    m=read(a.manifest)
    sys.path.insert(0,str(a.repository/"src"))
    env_manifest=environment_preflight(a,m)  # All imports/checks before GPU lock.
    root=a.output/"scheduler-v2/events"
    ledger=Ledger(root,m)
    existing=adopt_existing(a,ledger,write=False)
    print("DRY_RUN",json.dumps({"verified_completed":len(existing),
        "verified_successes":sum(x["state"]=="COMPLETED_SUCCESS" for x in existing.values()),
        "legacy_qwen_formal_trials":sum(t.startswith("r01-qwen") for t in existing),
        "pending_if_legacy_adopted":sum(x["state"]=="PENDING" for x in ledger.states.values())
                                  -sum(ledger.states[k]["state"]=="PENDING" for k in existing),
        "model_calls":0,"new_gpu_checks":0,
        "environment":env_manifest["module_identity"]}),flush=True)
    if not a.health_check and not a.execute:
        return 0
    if not a.allow_real_gpu:
        parser.error("explicit GPU approval required for health or benchmark")
    a.infra_output.mkdir(parents=True,exist_ok=True,mode=0o700)
    if a.health_check:
        name=uuid.uuid4().hex
        with a.gpu_lock.open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with server(a,43,a.infra_output/f"preflight-health-{name}.log") as health:
                proof={**env_manifest,"health":health,"manifest_sha256":MANIFEST_SHA}
                atomic_json(a.infra_output/f"runtime-manifest-{name}.json",proof)
                print("HEALTH_PROOF",str(a.infra_output/f"runtime-manifest-{name}.json"),flush=True)
        if not a.execute:
            return 0
    if not a.allow_paid_calls:
        parser.error("explicit paid call approval required")
    proofs=[read(x) for x in a.infra_output.glob("runtime-manifest-*.json")]
    assert any(x["environment_report_sha256"]==env_manifest["environment_report_sha256"]
               and x["pip_freeze_sha256"]==env_manifest["pip_freeze_sha256"]
               and x["serving_commands"]==env_manifest["serving_commands"]
               and x["health"]["status"]=="completed" for x in proofs)
    # Serialize recovery and all GPU/model task actions with the existing lock.
    with a.gpu_lock.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=Ledger(root,m)
        adopt_existing(a,ledger,write=True)
        if not any(x["kind"]=="INFRA_EVENT" and x["details"].get("legacy_qwen_startup")
                   for x in ledger.records):
            ledger.append("INFRA_EVENT",details={"legacy_qwen_startup":True,
                 "block":"r01-qwen","reason":"ModuleNotFoundError: typing_extensions",
                 "log_sha256":frozen.file_digest(a.output/"local-server-rep01.log"),
                 "formal_trials_reserved":0})
        ledger.append("SCHEDULER_STARTED",details={"environment_report_sha256":env_manifest["environment_report_sha256"],
                      "controller_commit":frozen.git(Path(__file__).resolve().parents[1],"rev-parse","HEAD")})
        progress(a,ledger)
        # Reconcile previously reserved trials without model resampling. Do not
        # launch more work if an external side effect remains uncertain.
        for tid,state in list(ledger.states.items()):
            if state["state"]=="RUNNING":
                try:
                    recovered=recover_running(a,ledger,tid)
                except Exception as exc:
                    ledger.terminal(tid,"OUTCOME_UNKNOWN",{"error_type":type(exc).__name__,"no_resample":True})
                    recovered="OUTCOME_UNKNOWN"
                print("RECONCILED",tid,recovered,flush=True)
                progress(a,ledger)
        if any(s["state"] in ("OUTCOME_UNKNOWN","INFRA_ABORTED") for s in ledger.states.values()):
            ledger.append("SCHEDULER_STOPPED",details={"reason":"unresolved or aborted reserved trial; no replacement"})
            return 2
        for block in m["blocks"]:
            pending=[tid for tid in block["trial_ids"] if ledger.states[tid]["state"]=="PENDING"]
            if not pending:
                continue
            frozen.require_frozen(a.repository,m)
            try:
                if block["model"]=="qwen":
                    startup=uuid.uuid4().hex
                    ledger.append("INFRA_EVENT",details={"event":"LOCAL_SERVER_START","block":block["trial_ids"][0],
                                 "startup_attempt_id":startup,"command_sha256":h(serve_argv(a,block["server_seed"]))})
                    context=server(a,block["server_seed"],a.infra_output/f"serve-{startup}.log")
                else:
                    context=contextlib.nullcontext(None)
                with context as health:
                    if health:
                        ledger.append("HEALTH_PASSED",details={"health":health,"startup_attempt_id":startup})
                    env=frozen.session_environment(a,block["model"])
                    for tid in pending:
                        try:
                            category=run_new(a,ledger,tid,env)
                        except Exception as exc:
                            if ledger.states[tid]["state"]=="RUNNING":
                                ledger.terminal(tid,"OUTCOME_UNKNOWN",{"error_type":type(exc).__name__,"no_resample":True})
                            print("TRIAL_INFRA_ERROR",tid,type(exc).__name__,str(exc)[:200],flush=True)
                            progress(a,ledger)
                            return 2
                        print("TRIAL_END",tid,category,flush=True)
                        progress(a,ledger)
                        if not category.startswith("COMPLETED_"):
                            return 2
            except Exception as exc:
                ledger.append("INFRA_EVENT",details={"event":"BLOCK_INFRA_ABORTED","block":block["trial_ids"][0],
                                "error_type":type(exc).__name__,"unreserved_trials":len(pending)})
                print("BLOCK_INFRA_ABORTED",block["trial_ids"][0],type(exc).__name__,str(exc)[:200],flush=True)
                progress(a,ledger)
                return 2
            for _ in range(30):
                if frozen.gpu_state()["memory_used_mib"]<1000:
                    break
                time.sleep(2)
            else:
                ledger.append("INFRA_EVENT",details={"event":"GPU_NOT_IDLE_AFTER_BLOCK"})
                return 2
        final=progress(a,ledger)
        ledger.append("SCHEDULER_STOPPED",details={"reason":"all scheduled units terminal",
                      "completed_model_trials":final["completed_model_trials"]})
        progress(a,ledger)
        return 0 if final["completed_model_trials"]==60 else 2


if __name__=="__main__":
    raise SystemExit(main())
