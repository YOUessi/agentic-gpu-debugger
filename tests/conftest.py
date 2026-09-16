import json

import pytest


@pytest.fixture
def store(tmp_path):
    from gpu_agent.store import RunStore

    return RunStore(tmp_path / "runs")


@pytest.fixture
def oob_service(store, tmp_path):
    import difflib
    import hashlib
    from pathlib import Path

    from gpu_agent.agent.models import (
        DiagnosisResult,
        EvidenceClaim,
        FinishAction,
        MemcheckAction,
        RetrieveDocsAction,
    )
    from gpu_agent.agent.provider import FakeProvider
    from gpu_agent.environment import RuntimeToolchainAttestation
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.models import SourceLocation
    from gpu_agent.execution.process import ProcessCapture
    from gpu_agent.knowledge.models import make_chunk
    from gpu_agent.knowledge.retrieve import KnowledgeIndex
    from gpu_agent.service import ApplicationService

    class FakeBackend(IsolatedGPUBackend):
        """Replace only the container subprocess; keep artifact/provenance plumbing."""

        def _attest_runtime(self):
            assert self._expected_toolchain is not None
            return RuntimeToolchainAttestation(
                runtime_session_id=self._runtime_session_id,
                lock_hash=self._expected_toolchain.lock_hash,
                image_id=self._expected_toolchain.image_id,
                cuda_nvcc=self._expected_toolchain.cuda_nvcc,
                compute_sanitizer=self._expected_toolchain.compute_sanitizer,
                compute_capability="8.9",
                target_arch=self._expected_toolchain.target_arch,
                policy_hash=hashlib.sha256(self.policy.model_dump_json().encode()).hexdigest(),
            )

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


@pytest.fixture
def test_schedule_commit_client(tmp_path):
    from schedule_authority_support import (
        TestScheduleCommitClient,
        register_test_schedule_client,
    )

    client = TestScheduleCommitClient.create(tmp_path / "test-schedule-authority")
    register_test_schedule_client(client)
    return client


@pytest.fixture
def native_evaluation_executor(
    oob_service, tmp_path, monkeypatch, request, test_schedule_commit_client
):
    """Fast native evaluation producer backed by the real store/orchestrator schemas."""
    import hashlib

    from gpu_agent.agent.prompts import PROMPT_VERSION
    from gpu_agent.benchmark.builder import BenchmarkBuilder
    from gpu_agent.benchmark.executor import EvaluationExecutor
    from gpu_agent.benchmark.ledger import CorpusFamily
    from gpu_agent.benchmark.models import (
        AuthoritativeCaseRegistry,
        AuthoritativeCaseSpec,
        CaseExecutionPlan,
    )
    from gpu_agent.benchmark.validation import CaseValidationController
    from gpu_agent.contracts import RepositorySnapshot, RunBinding
    from gpu_agent.environment import RuntimeToolchainAttestation, load_toolchain_lock
    from gpu_agent.execution.isolated import LOCK_PATH, IsolatedGPUBackend
    from gpu_agent.execution.process import ProcessCapture
    from gpu_agent.knowledge.models import make_chunk
    from gpu_agent.knowledge.retrieve import KnowledgeIndex
    from gpu_agent.store import RunStore

    service, provider, source = oob_service
    chunk = service.knowledge.chunks[0]
    fields = chunk.model_dump(exclude={"chunk_id", "content_hash", "text"})
    service.knowledge = KnowledgeIndex(
        [
            make_chunk(
                **fields,
                text=chunk.text + " Invalid __global__ write is a memcheck finding.",
            )
        ]
    )
    toolchain_hash = load_toolchain_lock(LOCK_PATH).lock_hash
    service._binding = RunBinding(
        repository=RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True),
        purpose="evaluation",
        toolchain_lock_hash=toolchain_hash,
        prompt_version=PROMPT_VERSION,
        model_config_hash="5" * 64,
    )
    requested_split = getattr(request, "param", "public")
    exact_verification_source = requested_split == "public_exact"
    split = "public" if exact_verification_source else requested_split
    if split not in {"public", "private"}:
        raise ValueError("invalid native evaluation fixture split")
    if exact_verification_source:
        import difflib

        exact_source = (
            __import__("pathlib").Path(__file__).resolve().parents[1]
            / "benchmarks/public/case_0001/public_input/kernel.cu"
        ).read_text()
        (source / "kernel.cu").write_text(exact_source)
        fixed_source = exact_source.replace(
            "out[i] = a[i] + b[i];", "if (i < n) out[i] = a[i] + b[i];"
        ).replace("n != 257", "n == 0")
        provider.diff = "".join(
            difflib.unified_diff(
                exact_source.splitlines(True),
                fixed_source.splitlines(True),
                fromfile="a/kernel.cu",
                tofile="b/kernel.cu",
            )
        )
    visibility = "public" if split == "public" else "evaluator"
    corpus = RunStore(tmp_path / "corpus", visibility=visibility)
    family = CorpusFamily._provision_for_test(
        tmp_path / "corpus-controller",
        public_store=corpus.root if visibility == "public" else tmp_path / "public-corpus",
        evaluator_store=corpus.root if visibility == "evaluator" else tmp_path / "evaluator-corpus",
        repository=tmp_path / "repository",
        schedule_public_key=test_schedule_commit_client.public_key,
    )
    monkeypatch.setenv("GPU_AGENT_CORPUS_FAMILY_ROOT", str(family.root))
    assert service.binding is not None
    service._binding = service.binding.model_copy(
        update={"corpus_ledger_namespace_hash": family.namespace_hash}
    )
    source_root = tmp_path / "corpus-sources"
    clean_root, mutant_root = source_root / "clean", source_root / "mutant"
    clean_root.mkdir(parents=True)
    mutant_root.mkdir(parents=True)
    mutant_bytes = (source / "kernel.cu").read_bytes()
    clean_bytes = mutant_bytes.replace(
        b"out[i] = a[i] + b[i];", b"if (i < n) out[i] = a[i] + b[i];"
    )
    (clean_root / "kernel.cu").write_bytes(clean_bytes)
    (mutant_root / "kernel.cu").write_bytes(mutant_bytes)
    harness_root = source_root / "harness"
    harness_root.mkdir()
    repository = __import__("pathlib").Path(__file__).resolve().parents[1]
    for name in ("vector_io.cpp", "vector_api.h", "json.hpp"):
        source_name = "vendor/json.hpp" if name == "json.hpp" else name
        (harness_root / name).write_bytes(
            (repository / "benchmarks/harness" / source_name).read_bytes()
        )
    input_bytes = json.dumps({"n": 257, "a": [1.0] * 257, "b": [2.0] * 257}).encode()
    harness_hash = hashlib.sha256(
        json.dumps(
            sorted(
                (name, hashlib.sha256((harness_root / name).read_bytes()).hexdigest())
                for name in ("vector_io.cpp", "vector_api.h", "json.hpp")
            ),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    spec = AuthoritativeCaseSpec(
        case_id="case_0100",
        template_id="vector-add",
        mutation_id="delete-guard",
        split=split,
        clean_source_hash=hashlib.sha256(clean_bytes).hexdigest(),
        mutant_source_hash=hashlib.sha256(mutant_bytes).hexdigest(),
        harness_hash=harness_hash,
        input_set_hash=hashlib.sha256(input_bytes).hexdigest(),
        oracle_id="vector-add-cpu-v1",
        target_tool="memcheck",
        expected_finding="Invalid __global__ write",
        sanitizer_repetitions=1,
        mutation_provenance_hash="9" * 64,
    )
    registry_bytes = AuthoritativeCaseRegistry(cases=[spec]).model_dump_json().encode()
    registry_hash = hashlib.sha256(registry_bytes).hexdigest()
    corpus_binding = RunBinding(
        repository=RepositorySnapshot(commit="a" * 40, tracked_tree_hash="b" * 64, clean=True),
        purpose="corpus_validation",
        toolchain_lock_hash=toolchain_hash,
        case_registry_hash=registry_hash,
        corpus_ledger_namespace_hash=family.namespace_hash,
    )

    class CorpusBackend(IsolatedGPUBackend):
        def _attest_runtime(self):
            assert self._expected_toolchain is not None
            return RuntimeToolchainAttestation(
                runtime_session_id=self._runtime_session_id,
                lock_hash=self._expected_toolchain.lock_hash,
                image_id=self._expected_toolchain.image_id,
                cuda_nvcc=self._expected_toolchain.cuda_nvcc,
                compute_sanitizer=self._expected_toolchain.compute_sanitizer,
                compute_capability="8.9",
                target_arch=self._expected_toolchain.target_arch,
                policy_hash=hashlib.sha256(self.policy.model_dump_json().encode()).hexdigest(),
            )

        def _container(self, path, operation, timeout, *, stdin=b"", cancel=None):
            mutant = b"if (i < n)" not in (path / "kernel.cu").read_bytes()
            if operation == "build":
                return ProcessCapture(0, b"", b"", False), b"binary", b""
            output = json.dumps(
                {"dtype": "float32", "shape": [257], "values": [3.0] * 257}
            ).encode()
            if operation == "run":
                return ProcessCapture(0, output, b"", False), b"", b""
            log = (
                b"========= Invalid __global__ write of size 4 bytes\n"
                b"=========     at kernel in kernel.cu:9\n"
                b"========= ERROR SUMMARY: 1 error\n"
                if mutant
                else b"========= ERROR SUMMARY: 0 errors\n"
            )
            return ProcessCapture(86 if mutant else 0, output, b"", False), b"", log

    backend = CorpusBackend(corpus, source_root, tmp_path / "corpus-tasks")
    controller = CaseValidationController._for_test(corpus, backend, corpus_binding, registry_bytes)

    def execute_case(role):
        names = [
            f"{role}/kernel.cu",
            *[f"harness/{name}" for name in ("vector_io.cpp", "vector_api.h", "json.hpp")],
        ]
        return controller.execute(
            CaseExecutionPlan(
                case_id="case_0100",
                template_id="vector-add",
                mutation_id="clean" if role == "clean" else "delete-guard",
                role=role,
                split=split,
                source_manifest={
                    name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                    for name in names
                },
                target_tool="memcheck",
                expected_finding="Invalid __global__ write",
                sanitizer_repetitions=1,
                case_registry_hash=registry_hash,
                case_spec_hash=controller.spec_hash(spec),
                mutation_provenance_hash=spec.mutation_provenance_hash,
            ),
            input_bytes,
        )

    clean_id, mutant_id = execute_case("clean"), execute_case("mutant")
    builder = BenchmarkBuilder(corpus)
    builder.register(builder.validate(clean_id, mutant_id))
    from gpu_agent.benchmark.schedule_authority import EvaluationScheduleVerifier

    return EvaluationExecutor(
        service,
        corpus,
        {"case_0100": source},
        _corpus_family=family,
        _schedule_verifier=EvaluationScheduleVerifier._for_test(family, service.store),
    )


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
