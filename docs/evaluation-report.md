# Recorded evaluation results — e80ce75

The evaluation implementation records five modes, three-or-more repeats, randomized serial
execution, separate denominators, blind views, latency, usage, and nullable cost. Offline
contract tests and live experiments are distinct evidence. The frozen commit
`e80ce75dde90e7624a89a63e6de1c68ef18bb7d5` completed both signed evaluations:

- Development: `1bc6af1b1acb4940a2c0829fc5d274aa`, 240/240 units.
- Holdout: `1a218a09a0414b459a732b8b9fd030bc`, 120/120 units.
- Failed holdout startup `60a614169aaa40c998497257c3fb3159` remains intact. It completed
  zero units because registered inputs were not adjacent to the kernels. The replacement
  used hash-identical, evaluator-only source/input snapshots; no failed record was overwritten.

Development VERIFIED_FIXED counts (48 units per mode): A 4, B 7, C 43, D 44, E 41.
Holdout VERIFIED_FIXED counts (24 units per mode): A 3, B 6, C 20, D 22, E 21.
These are raw verification counts, not independently adjudicated diagnosis scores. E does
not outperform D in these runs. Three repetitions per case are not independent cases.

The development set contains 139 fixed, 81 model-declared inconclusive, 11 NOT_FIXED,
six timeouts, two terminal output-contract rejections and one inconclusive verification.
All 643 physical provider calls were inventoried from their public records. Known development
cost is USD 2.986159 under the bound reviewed rate card; six units have unknown cost because
usage was unavailable. This is not a complete billing total. There are no dollar caps.

The holdout contains 72 fixed, nine NOT_FIXED and 39 units without a verification verdict,
including one timeout. All 120 native evidence chains were validated without exposing private
case contents. Independent manual scoring has not been supplied, and the original release
gate has not passed. The 18/18 supplemental GPU/isolation tests do not replace that gate.

The development-only inventory tool is `tools/dev_failure_inventory.py`. Raw logs, API keys,
private sources and evaluator judgments are not committed. Later engineering fixes remain
separate from this frozen baseline; passing regression tests does not relabel these results.
