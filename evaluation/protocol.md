# Evaluation protocol

All modes use the same diagnosis/patch provider, schemas, inputs, toolchain, RAG
index, and physical budgets. A/B/C compare fixed evidence; D/E compare evidence
acquisition policy. Pre-collected evidence reports collection and inference costs
separately. A provider failure in any mode remains that unit's failure record and is
never replaced by another mode; unit failures never stop the batch. Status, evidence
gate and release rules are specified in `docs/mode-contract.md`.

Each mode/case runs at least three times, serially on one GPU. Mode order is
randomized with a recorded seed. Timeouts remain in the end-to-end denominator.
Family/root/location metrics publish their separate valid denominators. Unknown
prices produce `cost=null`. Token usage and calculable costs are recorded only;
there are no dollar ceilings, balance checks, or cost-triggered batch stops.
