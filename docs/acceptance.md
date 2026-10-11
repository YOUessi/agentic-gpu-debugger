# Acceptance status

Frozen commit e80ce75 completed native registration of 16 public and eight private cases,
the signed 240-unit development evaluation and the signed 120-unit holdout evaluation.
See [recorded results](evaluation-report.md) for IDs, counts and failure denominators.

All 18 tests in the frozen GPU/isolation allowlist passed with zero skips in the supplemental
run. The exact collected IDs and imported source path were checked. The original collection
controller rejected spaces in parameterized test names before running any tests; that bug
is fixed in the development worktree, with a regression test using the actual allowlist.
The supplemental run is not misrepresented as a successful original controller run.

All 120 holdout native evidence chains, staged source/input hashes, consistent bindings and
the retained failed startup were checked. No model failure or unknown provider cost was erased.
Independent manual score labels and the full formal gate remain uncompleted; these do not
mean the 360-unit experiment is still waiting to run. Costs are record-only, never capped.

Later code changes have their own targeted regression tests. Test counts from a newer checkout
must not be copied into the frozen version's evidence. No rerun is implied by this status page.
