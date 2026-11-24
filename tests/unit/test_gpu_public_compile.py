"""Compile diagnostics expose only bounded candidate source information."""

from gpu_agent.benchmark.public_compile import public_compile_diagnostic


def test_nvcc_error_after_long_preamble() -> None:
    candidate = b"const int scale = 1;\nscale = 2;\n"
    stderr = b"/private/host/include/file.h: info\n" * 200 + (
        b"/input/kernel.cu(2): error: expression must be a modifiable lvalue\n      scale = 2;\n"
    )
    d = public_compile_diagnostic(stderr, candidate)
    assert d.line == 2
    assert d.source_excerpt == "scale = 2;"
    assert d.message == "expression must be a modifiable lvalue"
    assert "/private" not in d.model_dump_json()


def test_unrecognized_diagnostic_does_not_expose_host_text() -> None:
    d = public_compile_diagnostic(b"/private/token: arbitrary error", b"int x;\n")
    assert d.line is None
    assert "/private" not in d.model_dump_json()
