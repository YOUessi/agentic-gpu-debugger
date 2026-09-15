# Evaluation protocol

All modes use the same diagnosis/patch provider, schemas, inputs, toolchain, RAG
index, and physical budgets. A/B/C compare fixed evidence; D/E compare evidence
acquisition policy. Pre-collected evidence reports collection and inference costs
separately. Provider failure in E remains an E failure and is never replaced by D.

Each mode/case runs at least three times, serially on one GPU. Mode order is
randomized with a recorded seed. Timeouts remain in the end-to-end denominator.
Family/root/location metrics publish their separate valid denominators. Unknown
prices produce `cost=null`; an absent or exceeded cost cap stops the batch while
preserving completed records.
