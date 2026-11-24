# h003 independent Strict Fix — 2026-11-24

Frozen Agent implementation: `1e825aa43d51eccf05c8b5953d5705d0554e1c65`. Original conditional-recovery trial: A-build-h003, public Agent run `d0b379c4fa064a37b10318459a1d21d4`, public Checker child `4324fb4c87e443ee86ec685beff6ce6f`.

The operator verified the closed original session with the existing `harness verify-repo` command. Preconditions checked terminal journal, accepted worktree hashes, exact final diff, passing Checker child provenance, registry mutant identity, and exact candidate materialization. No new patch, model call, or candidate selection occurred. Candidate SHA256 `992a461c7f14930797781d15be985706ce5eb4a57215f82a98c73c4194861c2e`; final diff SHA256 `feae08c086669a12d85c5b1a6e0b259c6b7dfe0299dc7e52e083d85d06d8bc8b`.

Independent real strict evaluator returned:

```json
{
  "schema_version": "hardset-experimental-strict-v2",
  "case_id": "h003",
  "candidate_sha256": "992a461c7f14930797781d15be985706ce5eb4a57215f82a98c73c4194861c2e",
  "verdict": "VERIFIED_FIXED",
  "reason_code": "ALL_REQUIRED_CHECKS_PASSED"
}
```

Operator verifier exit status 0; strict public projection run ID `75198510cf60437dbe200fdd84598996`. The projection file SHA256 is `ef0485256f6fc19c45caafab137ec5d454f0f6c60e820e731af17d8389ef2e76`. Local public projection is at `/home/you/gpu-native-harness-validation-20261124/p4-strict-and-binding/h003-strict-projection.json`. Private evaluator evidence remains segregated on the authorized machine and is not included here or in model requests. The original public Agent result retains its historical DIFF_STAGED_UNVERIFIED status; independent verification is an append-only separate projection bound to the same candidate, not a rewrite of the earlier result.

This is one genuine end-to-end conditional recovery followed by independent Strict Fix on the same candidate. It does not establish population-level repair reliability, generalization, or a causal benefit of rollback.
