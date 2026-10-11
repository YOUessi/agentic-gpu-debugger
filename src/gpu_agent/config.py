"""Operator-controlled paths. Never infer a compiler from a PyTorch runtime."""

import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cuda_root: Path = Field(default_factory=lambda: Path(sys.prefix))
    cuda_bin: Path | None = None
    host_compiler: Path = Path("/usr/bin/g++")
    nvidia_smi: Path = Path("/usr/bin/nvidia-smi")
    target_arch: str = Field(default="sm_89", pattern=r"^sm_[0-9]+$")
    probe_timeout_seconds: float = Field(default=5, gt=0, le=30)

    @field_validator("cuda_root", "cuda_bin", "host_compiler", "nvidia_smi")
    @classmethod
    def absolute_path(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ValueError("toolchain paths must be absolute")
        return value

    @property
    def bin_dir(self) -> Path:
        return self.cuda_bin if self.cuda_bin is not None else self.cuda_root / "bin"
