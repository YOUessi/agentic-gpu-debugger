"""RunStore is the only persistence layer; every nested reference is checked."""

from pydantic import BaseModel

from gpu_agent.contracts import ArtifactRef
from gpu_agent.evidence.models import EvidenceBundle
from gpu_agent.store import RunStore


class EvidenceRepository:
    def __init__(self, store: RunStore, *, evaluator: bool = False) -> None:
        if store.visibility != ("evaluator" if evaluator else "public"):
            raise ValueError("public evidence requires a public store")
        self._store = store

    def _validate(self, run_id: str, value: object) -> None:
        if isinstance(value, ArtifactRef):
            if value.run_id != run_id or value.visibility != self._store.visibility:
                raise ValueError("evidence reference crosses run or visibility boundary")
            self._store.read(value)
        elif isinstance(value, BaseModel):
            for name in type(value).model_fields:
                self._validate(run_id, getattr(value, name))
        elif isinstance(value, list):
            for item in value:
                self._validate(run_id, item)
        elif isinstance(value, dict):
            for item in value.values():
                self._validate(run_id, item)

    def save(self, run_id: str, bundle: EvidenceBundle) -> ArtifactRef:
        self._validate(run_id, bundle)
        return self._store.put(
            run_id,
            "evidence/bundle.json",
            bundle.model_dump_json().encode(),
            self._store.visibility,
        )

    def public_view(self, run_id: str) -> EvidenceBundle:
        if self._store.visibility != "public":
            raise ValueError("evaluator evidence has no public view")
        return self.view(run_id)

    def view(self, run_id: str) -> EvidenceBundle:
        """Controller-only evidence in this repository's single visibility domain."""
        manifest = self._store.load(run_id)
        refs = [ref for ref in manifest.artifact_refs if ref.name == "evidence/bundle.json"]
        bundle = (
            EvidenceBundle.model_validate_json(self._store.read(refs[-1]))
            if refs
            else (EvidenceBundle())
        )
        self._validate(run_id, bundle)
        return bundle
