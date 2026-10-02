# Changelog

All notable changes are documented here. This project follows Semantic
Versioning and the structure of Keep a Changelog.

## [Unreleased]

### Added

- React/TypeScript CUDA Test & Repair Console with server-side run filtering,
  iterative repair timeline, diff/evidence inspection, and strict-verification actions.
- Optional FastAPI/uvicorn web adapter that reuses `ApplicationService` and the public
  RunStore instead of duplicating Agent or verifier logic.
- Vitest and Playwright coverage for the operator workflow, plus FastAPI tests backed by
  a temporary real RunStore.

### Security

- Web repair requests resolve allowlisted public case IDs instead of arbitrary host paths.
- Artifact reads are limited to registered public RunStore refs with bounded response size.
- Paid model calls remain explicit opt-in and are still constrained by controller policy.

- Final V2 release evidence and the frozen paid evaluation remain gated by the
  repository's acceptance policy.

## [0.2.0] - 2026-09-20

### Added

- Evidence-driven CUDA diagnosis and single-candidate verification.
- Isolated Docker GPU execution with four Compute Sanitizer tools.
- Public/private corpus controls, immutable evidence, and release gates.
- Public seed batch validation, reporting, and bounded evidence export.
- Reproducible source and wheel packaging with installed runtime resources.

### Security

- Candidate workloads run with a locked container policy and no network.
- Provider credentials are excluded from candidate execution and evidence.

[Unreleased]: https://github.com/YOUessi/agentic-gpu-debugger/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/YOUessi/agentic-gpu-debugger/releases/tag/v0.2.0
