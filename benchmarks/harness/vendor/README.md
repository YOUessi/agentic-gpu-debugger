# Vendored JSON parser

- Project: JSON for Modern C++ by Niels Lohmann (`nlohmann/json`).
- Version: **3.12.0**, upstream tag `v3.12.0` (released 2025-04-11).
- Release: https://github.com/nlohmann/json/releases/tag/v3.12.0
- Unmodified single header: https://raw.githubusercontent.com/nlohmann/json/v3.12.0/single_include/nlohmann/json.hpp
- `json.hpp` SHA-256, verified against the release's published checksum:
  `aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63`
- License: MIT; see `LICENSE.MIT` and the notices retained in `json.hpp`.
- License source: https://raw.githubusercontent.com/nlohmann/json/v3.12.0/LICENSE.MIT

The trusted harness uses strict whole-document JSON parsing, rejects duplicate
keys, constrains nesting to the vector protocol, and validates numeric types,
array lengths, finite float32 range, and a 4 MiB input limit before execution.
Ordinary JSON numbers are rounded to float32; exact decimal representability
is not required. Output is encoded only after all values pass finiteness checks.
The parser and this harness must be included in the reviewed source manifest.
