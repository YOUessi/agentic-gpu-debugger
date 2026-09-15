# Evaluation status

The evaluation implementation records five modes, three-or-more repeats, randomized serial
execution, separate denominators, blind views, latency, usage, and nullable cost. Offline
contract tests pass. The full 24×5×3 = 360-unit DeepSeek experiment has not run because no
API cost ceiling has been configured. No quality comparison or uplift is claimed.

Release validation additionally requires one complete development run and one complete
holdout run, each with all A–E modes and every case repeated at least three times. Schedule,
attempt and record artifacts are re-read and hash checked; commit, prompt, toolchain, model
configuration and schedule hashes must agree. Partial or stopped batches are not release
evidence even though they remain useful diagnostic data.

Private holdout identifiers are a remaining production gap. Persisting a private template or
operator name in `PublicEvaluationRecord` would leak the withheld split. The release loader
therefore accepts holdout evidence only when public schedule/records use non-identifying
aliases and an evaluator-only, same-commit identity map proves complete coverage. The current
production runner does not generate that map, so the paid holdout evaluation must not start
until that path is implemented and reviewed.
