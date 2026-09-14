import pytest


@pytest.fixture
def store(tmp_path):
    from gpu_agent.store import RunStore

    return RunStore(tmp_path / "runs")


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
