"""Offline stand-in for the container runtime used by verification."""

import hashlib
import json


def install_offline_container_boundary(monkeypatch):
    """Replace only the Docker boundary of IsolatedGPUBackend with a CPU oracle model.

    Subclasses that override `_container`/`_attest_runtime` (diagnosis fakes) are unaffected;
    the verification engine's own backend runs this deterministic vector-add model.
    """
    from gpu_agent.environment import RuntimeToolchainAttestation
    from gpu_agent.execution.isolated import IsolatedGPUBackend
    from gpu_agent.execution.process import ProcessCapture

    calls = []

    def attest(self):
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

    def container(self, path, operation, timeout, *, stdin=b"", cancel=None):
        source = (path / "kernel.cu").read_text()
        assert set(p.name for p in path.iterdir()) <= {
            "kernel.cu",
            "vector_io.cpp",
            "vector_api.h",
            "json.hpp",
            "vector_add",
        }
        if operation == "build":
            assert not ((path / "kernel.cu").stat().st_mode & 0o222)
            if "INVALID CUDA SYNTAX" in source:
                return ProcessCapture(1, b"", b"compiler error", False), b"", b""
            return ProcessCapture(0, b"", b"", False), hashlib.sha256(source.encode()).digest(), b""
        data = json.loads(stdin)
        assert set(data) == {"a", "b", "n"}
        calls.append((path, operation, data))
        values = [a + b for a, b in zip(data["a"], data["b"], strict=True)]
        if "0.0f" in source and ("n == 257 ?" not in source or data["n"] != 257):
            values = [0] * data["n"]
        output = json.dumps({"dtype": "float32", "shape": [data["n"]], "values": values}).encode()
        if operation == "memcheck" and "if (i < n)" not in source:
            log = (
                b"========= Invalid __global__ read of size 4 bytes\n"
                b"=========     at vector_add(float const *, float const *, float *, unsigned long)"
                b" in /input/kernel.cu:10\n========= ERROR SUMMARY: 1 error\n"
            )
            return ProcessCapture(86, output, b"", False), b"", log
        clean = (
            b"========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"
            if operation == "racecheck"
            else b"========= ERROR SUMMARY: 0 errors\n"
        )
        return ProcessCapture(0, output, b"", False), b"", clean

    monkeypatch.setattr(IsolatedGPUBackend, "_container", container)
    monkeypatch.setattr(IsolatedGPUBackend, "_attest_runtime", attest)
    return calls
