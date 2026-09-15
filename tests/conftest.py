import pytest


@pytest.fixture
def store(tmp_path):
    from gpu_agent.store import RunStore

    return RunStore(tmp_path / "runs")


@pytest.fixture
def oob_service(store, tmp_path):
    import difflib
    from pathlib import Path

    from gpu_agent.agent.models import (
        DiagnosisResult,
        EvidenceClaim,
        FinishAction,
        MemcheckAction,
        RetrieveDocsAction,
    )
    from gpu_agent.agent.provider import FakeProvider
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.models import SourceLocation
    from gpu_agent.execution.process import ProcessCapture
    from gpu_agent.knowledge.models import make_chunk
    from gpu_agent.knowledge.retrieve import KnowledgeIndex
    from gpu_agent.service import ApplicationService

    class FakeBackend(IsolatedGPUBackend):
        """Replace only the container subprocess; keep artifact/provenance plumbing."""

        def _container(self, path, operation, timeout, *, stdin=b"", cancel=None):
            if operation in {"build", "build_standalone"}:
                return ProcessCapture(0, b"", b"", False), b"fake binary", b""
            if operation == "run":
                return ProcessCapture(0, b'{"values":[3]}', b"", False), b"", b""
            log = (
                b"========= COMPUTE-SANITIZER\n"
                b"========= Invalid __global__ write of size 4 bytes\n"
                b"=========     at vector_add(float const *, float const *, float *, unsigned long)"
                b" in /input/kernel.cu:9\n"
                b"=========     by thread (257,0,0) in block (1,0,0)\n"
                b"========= ERROR SUMMARY: 1 error\n"
            )
            return ProcessCapture(86, b'{"values":[3]}', b"", False), b"", log

    class OOBFakeProvider(FakeProvider):
        """Bind this test's scripted claims to the run's freshly generated citation IDs."""

        forge_citations = False

        def diagnose(self, evidence):
            self.result = DiagnosisResult(
                diagnostic_outcome="DIAGNOSED",
                failure_family="out_of_bounds",
                root_cause="The thread index can exceed the input length.",
                source_locations=[SourceLocation(path="kernel.cu", line=9)],
                observed_facts=evidence.observed_facts,
                tool_findings=[
                    EvidenceClaim(text=f.category, citation_ids=[f.artifact_id])
                    for f in evidence.tool_findings
                ],
                documentation_evidence=[
                    EvidenceClaim(
                        text=d.text, citation_ids=["forged" if self.forge_citations else d.chunk_id]
                    )
                    for d in evidence.documentation
                ],
                model_inferences=["An index guard may prevent the reported write."],
                recommended_change="Guard the write with i < n.",
                confidence_label="high",
            )
            return super().diagnose(evidence)

    repo = Path(__file__).resolve().parents[1]
    original_path = repo / "benchmarks/public/case_0001/public_input/kernel.cu"
    original = "".join(
        line
        for line in original_path.read_text().splitlines(True)
        if "Intentional benchmark defect" not in line
    )
    source = tmp_path / "public_input"
    source.mkdir()
    (source / "kernel.cu").write_text(original)
    # Synthetic adjacent private files prove that input discovery does not sweep directories.
    (source / "reference.cu").write_text("secret-canary fixed reference")
    (source / "case.json").write_text('{"ground_truth":"secret-canary","private_seed":42}')
    (source / "checker.py").write_text("evaluation_label = 'secret-canary'\n")
    fixed = original.replace("out[i] = a[i] + b[i];", "if (i < n) out[i] = a[i] + b[i];").replace(
        "n != 257", "n == 0"
    )
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(True),
            fixed.splitlines(True),
            fromfile="a/kernel.cu",
            tofile="b/kernel.cu",
        )
    )
    chunk = make_chunk(
        source_id="memcheck",
        document_title="Compute Sanitizer",
        document_version="13.0",
        section_title="Memcheck",
        source_url="https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html",
        retrieved_at="2026-09-15T00:00:00Z",
        text="Memcheck detects out of bounds global memory writes.",
        block_ordinal=0,
        compatibility={"cuda": ">=13,<14", "compute_sanitizer": ">=13,<14"},
    )
    provider = OOBFakeProvider(
        [
            MemcheckAction(),
            RetrieveDocsAction(typed_arguments={"query": "out of bounds", "k": 3}),
            FinishAction(),
        ],
        DiagnosisResult.inconclusive("TEST_NOT_RUN"),
        diff,
    )
    service = ApplicationService(
        store,
        tmp_path / "evaluator",
        provider=provider,
        backend_factory=FakeBackend,
        knowledge=KnowledgeIndex([chunk]),
        knowledge_version="cuda=13.0;compute-sanitizer=13.0",
    )
    return service, provider, source


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--gpu-run-root",
        default=None,
        help="Optional artifact directory for real GPU acceptance runs (unique run IDs).",
    )
    parser.addoption(
        "--require-live",
        action="store_true",
        default=False,
        help="Treat skipped GPU/container/provider/release tests as failures.",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    outcome = yield
    report = outcome.get_result()
    # Module-level skips occur before markers can be inspected: fail closed.
    if collector.config.getoption("--require-live") and report.skipped:
        report.outcome = "failed"
        report.longrepr = "Required live collection was skipped; coverage cannot be established."


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    required = {"gpu", "container", "live_llm", "release"}
    if (
        item.config.getoption("--require-live")
        and report.skipped
        and any(item.get_closest_marker(mark) for mark in required)
    ):
        report.outcome = "failed"
        report.longrepr = "Required live test was skipped; this is not acceptance evidence."
