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
- Public case catalog across the core/diverse registries, live polling for active runs, and
  click-through diagnosis citations that resolve both RunStore artifacts and RAG chunk IDs.
- A persistent full-stack development log that records reproduced issues, confirmed causes,
  fixes, validation, and remaining boundaries across development rounds.
- Persistent asynchronous repair jobs with immediate HTTP 202 responses, controller run binding,
  restart recovery, and live RunManifest pipeline timelines.
- Original-source / selected-candidate diff review without reimplementing the controller patcher
  in the browser.
- Read-only Batch & Evaluation Analytics for public seed batches and A–E evaluation manifests,
  including mode comparison, large record grids, failure-family distribution, latency, model/tool
  usage, token counts, and known public costs.
- Optional `GPU_AGENT_ANALYTICS_RUN_ROOT` for mounting a historical public RunStore without
  changing the operational RunStore or exposing evaluator/private evidence.
- Descriptive mode charts for verified rate, latency, and model-call volume, plus immutable
  evaluation-record lineage drill-down into historical public diagnosis/evidence/candidate/
  verification runs without enabling operational write actions.
- Evaluation history timeline and compatibility-gated cross-run regression comparison using stable
  `(case_id, template_id, mode, repeat)` unit keys rather than schedule ordinals.
- Full public Analytics projection export as JSON/CSV, with bounded record counts and CSV formula
  injection protection for spreadsheet consumers.

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
