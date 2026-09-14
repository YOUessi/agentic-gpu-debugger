"""Public evidence rejects cross-run and evaluator references at the read boundary."""

import pytest


def test_public_view_roundtrip_and_scope(store):
    from gpu_agent.evidence.models import EvidenceBundle
    from gpu_agent.evidence.repository import EvidenceRepository

    run = store.create_run("unit")
    ref = store.put(run.id, "sources/kernel.cu", b"source", "public")
    repository = EvidenceRepository(store)
    bundle = EvidenceBundle(source_snapshot=[ref], limitations=["synthetic unit input"])
    repository.save(run.id, bundle)
    assert repository.public_view(run.id) == bundle
    other = store.create_run("other")
    with pytest.raises(ValueError, match="run"):
        repository.save(other.id, bundle)
    forged = bundle.model_copy(
        update={"source_snapshot": [ref.model_copy(update={"visibility": "evaluator"})]}
    )
    with pytest.raises(ValueError):
        repository.save(run.id, forged)


def test_public_view_validates_persisted_nested_refs(store):
    from gpu_agent.evidence.models import EvidenceBundle
    from gpu_agent.evidence.repository import EvidenceRepository

    run, sibling = store.create_run("unit"), store.create_run("sibling")
    secret = store.put(sibling.id, "secret", b"not visible", "public")
    bundle = EvidenceBundle(retrieved_chunks=[secret])
    store.put(run.id, "evidence/bundle.json", bundle.model_dump_json().encode(), "public")
    with pytest.raises(ValueError, match="run"):
        EvidenceRepository(store).public_view(run.id)
