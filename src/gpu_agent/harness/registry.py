"""Controller-owned tool registry and permission policy.

A model gets tool names and schemas; policy independently decides whether the
requested capability may run. No model-provided permission elevation is honored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Permission = Literal["ALLOW", "DENY", "APPROVAL_REQUIRED"]
ToolRisk = Literal["read", "gpu", "edit", "finish"]


@dataclass(frozen=True)
class ToolDescriptor:
    name: str
    risk: ToolRisk
    description: str


TOOL_REGISTRY: tuple[ToolDescriptor, ...] = (
    ToolDescriptor("list_files", "read", "List registered source files"),
    ToolDescriptor("read_file", "read", "Read bounded source lines"),
    ToolDescriptor("search_code", "read", "Search current kernel source"),
    ToolDescriptor("functional_probe", "gpu", "Run one bounded GPU input"),
    ToolDescriptor("apply_patch", "edit", "Stage a source-scoped unified diff"),
    ToolDescriptor("replace_snippet", "edit", "Make an exact, uniquely matched source edit"),
    ToolDescriptor("run_public_checks", "gpu", "Build and check the staged CUDA candidate"),
    ToolDescriptor("finish", "finish", "Finish only after public checks pass"),
    ToolDescriptor("repo_list_files", "read", "List frozen Git worktree files"),
    ToolDescriptor("repo_read_file", "read", "Read an allowed Git worktree file"),
    ToolDescriptor("repo_replace_snippet", "edit", "Exact edit of allowed source file"),
    ToolDescriptor(
        "repo_replace_many",
        "edit",
        "Atomically stage 2-4 disjoint exact replacements in one source file",
    ),
    ToolDescriptor("repo_git_diff", "read", "Inspect bounded Git changes"),
    ToolDescriptor(
        "repo_inspect_syntax",
        "read",
        "Inspect SHA-pinned CUDA AST statements with immutable node IDs",
    ),
    ToolDescriptor(
        "repo_replace_syntax",
        "edit",
        "Atomically rewrite CUDA syntax nodes using source SHA and parser validation",
    ),
    ToolDescriptor(
        "repo_probe_public",
        "gpu",
        "Public fixed-size functional probe on registered mutant or staged candidate",
    ),
    ToolDescriptor("repo_compile_candidate", "gpu", "Compile the exact public candidate only"),
    ToolDescriptor("repo_revert_last_edit", "edit", "Undo the most recent accepted AST edit once"),
    ToolDescriptor(
        "repo_run_public_checks",
        "gpu",
        "Run real isolated CUDA checks for one operator-registered public case",
    ),
)
_NAMES = {tool.name: tool for tool in TOOL_REGISTRY}


@dataclass(frozen=True)
class ToolPermissionPolicy:
    """Explicit operator privileges for a workspace session.

    APPROVAL_REQUIRED is a *hard stop* until an external operator grants it.
    The model cannot approve its own action by changing action arguments.
    """

    allow_gpu: bool = True
    allow_edit: bool = True
    denied_tools: frozenset[str] = field(default_factory=frozenset)
    approval_required: frozenset[str] = field(default_factory=frozenset)
    approved_tools: frozenset[str] = field(default_factory=frozenset)

    def check(self, tool: str) -> Permission:
        spec = _NAMES.get(tool)
        if spec is None or tool in self.denied_tools:
            return "DENY"
        if spec.risk == "gpu" and not self.allow_gpu:
            return "DENY"
        if spec.risk == "edit" and not self.allow_edit:
            return "DENY"
        if tool in self.approval_required and tool not in self.approved_tools:
            return "APPROVAL_REQUIRED"
        return "ALLOW"

    @property
    def available_tools(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in TOOL_REGISTRY if self.check(spec.name) == "ALLOW")

    def manifest(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {
                "name": spec.name,
                "risk": spec.risk,
                "description": spec.description,
                "permission": self.check(spec.name),
            }
            for spec in TOOL_REGISTRY
        )


def registered_tool_names() -> tuple[str, ...]:
    return tuple(tool.name for tool in TOOL_REGISTRY)
