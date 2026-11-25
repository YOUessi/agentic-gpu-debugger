"""Regression from the real two-file DeepSeek smoke at bf9e224."""

from gpu_agent.harness.graph_plan import GraphTaskPlan, evaluate_tasks
from gpu_agent.harness.graph_schedule import task_fingerprint


def test_helper_inspection_does_not_reopen_completed_prerequisites():
    for helper_path in ("bias.cu", ""):
        _assert_helper_preserves_prerequisites(helper_path)


def _assert_helper_preserves_prerequisites(helper_path):
    plan = GraphTaskPlan.model_validate(
        {
            "tasks": [
                {
                    "task_id": "t1",
                    "gate": "inspect",
                    "target": "",
                    "objective": "List the repository source files",
                    "depends_on": [],
                },
                {
                    "task_id": "t2",
                    "gate": "inspect",
                    "target": "",
                    "objective": "Read current kernel.cu and bias.cu source",
                    "depends_on": ["t1"],
                },
                {
                    "task_id": "t3",
                    "gate": "edit",
                    "target": "kernel.cu",
                    "objective": "Correct the kernel arithmetic",
                    "depends_on": ["t2"],
                },
                {
                    "task_id": "t4",
                    "gate": "edit",
                    "target": "bias.cu",
                    "objective": "Correct the bias arithmetic",
                    "depends_on": ["t2"],
                },
                {
                    "task_id": "t5",
                    "gate": "finish",
                    "target": "",
                    "objective": "Finish both source changes",
                    "depends_on": ["t3", "t4"],
                },
            ],
            "revision_reason": "Apply the requested two source edits",
            "failure_risk": "Do not repeat already completed source changes",
        }
    )
    keys = [task_fingerprint(task) for task in plan.tasks]
    evidence = [
        {
            "turn": 1,
            "gate": "inspect",
            "status": "OK",
            "path": "",
            "candidate": None,
            "task_key": keys[0],
        },
        {
            "turn": 2,
            "gate": "inspect",
            "status": "OK",
            "path": "kernel.cu",
            "candidate": None,
            "task_key": keys[1],
        },
        {
            "turn": 3,
            "gate": "edit",
            "status": "OK",
            "path": "kernel.cu",
            "candidate": "candidate-a",
            "task_key": keys[2],
        },
    ]
    snapshot = {"current_candidate_sha256": "candidate-a"}
    before = evaluate_tasks(plan, evidence, snapshot, ["kernel.cu"])
    assert [row["status"] for row in before] == ["DONE", "DONE", "DONE", "READY", "BLOCKED"]

    # A helper read/diff for t4 has no primary task key. It may support an
    # unfinished task, but must not replace earlier valid prerequisite proof.
    evidence.append(
        {
            "turn": 4,
            "gate": "inspect",
            "status": "OK",
            "path": helper_path,
            "candidate": "candidate-a",
            "task_key": None,
        }
    )
    after = evaluate_tasks(plan, evidence, snapshot, ["kernel.cu"])
    assert after == before

    evidence.append(
        {
            "turn": 5,
            "gate": "edit",
            "status": "OK",
            "path": "bias.cu",
            "candidate": "candidate-b",
            "task_key": keys[3],
        }
    )
    completed = evaluate_tasks(
        plan, evidence, {"current_candidate_sha256": "candidate-b"}, ["kernel.cu", "bias.cu"]
    )
    assert [row["status"] for row in completed] == ["DONE", "DONE", "DONE", "DONE", "READY"]
    assert [row["evidence_turn"] for row in completed] == [1, 2, 3, 5, None]
