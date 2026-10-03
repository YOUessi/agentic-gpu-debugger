# Security Policy

## Supported versions

Security fixes are applied to the latest release line. Development branches
and untagged source snapshots are not supported releases.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting flow under the repository's
Security tab. Do not open a public issue for a suspected vulnerability and do
not include API keys, private corpus data, evaluator evidence, or exploit data
in public logs.

Include the affected version or commit, a minimal reproduction, expected and
observed behavior, and the impact. Maintainers will acknowledge a complete
report through GitHub's advisory channel and coordinate disclosure there.

## Security boundary

Candidate code is isolated in a restricted Docker GPU container, but containers
share the host kernel and GPU driver. The project does not claim VM-grade
isolation against a hostile process running as the same host OS user. See
`docs/limitations.md` for the complete boundary.
