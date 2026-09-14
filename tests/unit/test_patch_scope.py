import difflib
import hashlib

import pytest


@pytest.fixture
def snapshot(tmp_path):
    from gpu_agent.patching import SourceSnapshot

    source = b"int value = 1;\n"
    (tmp_path / "kernel.cu").write_bytes(source)
    return SourceSnapshot(
        parent_run_id="a" * 32,
        root=tmp_path,
        hashes={"kernel.cu": hashlib.sha256(source).hexdigest()},
    )


GOOD = "--- a/kernel.cu\n+++ b/kernel.cu\n@@ -1 +1 @@\n-int value = 1;\n+int value = 2;\n"


@pytest.mark.parametrize(
    "added",
    [
        '// /*\n#include "/etc/passwd"\n// */\n',
        '// /* ignored opener\n%:include "/etc/passwd"\n// */\n',
        'const char* text = "/*";\n#include "/etc/passwd"\n// */\n',
        '// /*\n#inc\\\nlude "/etc/passwd"\n// */\n',
        '/* benign */ #include "/etc/passwd"\n',
        '/* line one\nline two */ #include "/etc/passwd"\n',
    ],
)
def test_comment_and_literal_order_cannot_hide_active_include(snapshot, added):
    from gpu_agent.patching import apply_candidate

    before = (snapshot.root / "kernel.cu").read_text()
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(True),
            (before + added).splitlines(True),
            fromfile="a/kernel.cu",
            tofile="b/kernel.cu",
        )
    )
    with pytest.raises(ValueError):
        apply_candidate(snapshot, diff, ["kernel.cu"])


def test_git_index_hashes_are_checked(snapshot):
    from gpu_agent.patching import apply_candidate

    with pytest.raises(ValueError):
        apply_candidate(
            snapshot,
            "diff --git a/kernel.cu b/kernel.cu\nindex 0000000..1111111 100644\n" + GOOD,
            ["kernel.cu"],
        )


def test_line_mapping_handles_insertions_and_deletions(snapshot):
    from gpu_agent.patching import apply_candidate, candidate_line_map, materialize_candidate

    diff = "--- a/kernel.cu\n+++ b/kernel.cu\n@@ -0,0 +1 @@\n+// added\n"
    candidate = apply_candidate(snapshot, diff, ["kernel.cu"])
    assert candidate_line_map(snapshot, candidate) == {1: 2}
    assert materialize_candidate(snapshot, candidate)["kernel.cu"] == b"// added\nint value = 1;\n"


def test_exact_patch_is_auditable_and_does_not_mutate_base(snapshot):
    from gpu_agent.patching import apply_candidate, materialize_candidate

    candidate = apply_candidate(snapshot, GOOD, ["kernel.cu"])
    assert candidate.generated_by == "human"
    assert candidate.scope_validation == "VALID"
    assert materialize_candidate(snapshot, candidate)["kernel.cu"] == b"int value = 2;\n"
    assert (snapshot.root / "kernel.cu").read_bytes() == b"int value = 1;\n"
    with pytest.raises(ValueError):
        materialize_candidate(
            snapshot, candidate.model_copy(update={"patched_source_hash": "0" * 64})
        )


@pytest.mark.parametrize(
    "diff",
    [
        GOOD.replace("kernel.cu", "/kernel.cu"),
        GOOD.replace("kernel.cu", "../kernel.cu"),
        GOOD.replace("kernel.cu", "vector_io.cpp"),
        GOOD.replace("+++ b/kernel.cu", "+++ b/other.cu"),
        GOOD.replace("--- a/kernel.cu", "--- /dev/null"),
        GOOD.replace("@@ -1 +1 @@", "@@ -2 +2 @@"),
        GOOD.replace("int value = 1;", "int value = 9;"),
        GOOD + "rename from kernel.cu\nrename to other.cu\n",
        "GIT binary patch\nliteral 4\n",
        GOOD.replace("+int value = 2;", '+#include "/etc/passwd"'),
        GOOD.replace("+int value = 2;", '+#include "../private.h"'),
        GOOD.replace("+int value = 2;", "+#include HEADER"),
        GOOD + GOOD,
    ],
)
def test_reject_out_of_scope_or_inexact_patch(snapshot, diff):
    from gpu_agent.patching import apply_candidate

    with pytest.raises(ValueError):
        apply_candidate(snapshot, diff, ["kernel.cu"])


def test_base_hash_symlink_and_allowlist_are_enforced(snapshot, tmp_path):
    from gpu_agent.patching import apply_candidate

    with pytest.raises(ValueError):
        apply_candidate(snapshot, GOOD, ["kernel.cu", "vector_io.cpp"])
    (tmp_path / "kernel.cu").write_text("tampered\n")
    with pytest.raises(ValueError):
        apply_candidate(snapshot, GOOD, ["kernel.cu"])
    (tmp_path / "kernel.cu").unlink()
    (tmp_path / "kernel.cu").symlink_to(tmp_path / "other")
    with pytest.raises(ValueError):
        apply_candidate(snapshot, GOOD, ["kernel.cu"])
