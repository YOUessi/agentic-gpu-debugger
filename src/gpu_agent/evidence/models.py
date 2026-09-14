"""Spec §8.3: facts and their sources only, with no diagnosis fields."""

from pydantic import Field

from gpu_agent.contracts import ArtifactRef
from gpu_agent.execution.models import (
    BuildResult,
    ExecutionModel,
    ExecutionResult,
    SanitizerResult,
    SourceLocation,
)


class EvidenceBundle(ExecutionModel):
    environment: dict[str, str] = Field(default_factory=dict)
    source_snapshot: list[ArtifactRef] = Field(default_factory=list)
    build_result: BuildResult | None = None
    execution_result: ExecutionResult | None = None
    sanitizer_results: list[SanitizerResult] = Field(default_factory=list)
    source_locations: list[SourceLocation] = Field(default_factory=list)
    retrieved_chunks: list[ArtifactRef] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
