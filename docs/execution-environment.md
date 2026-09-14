# Isolated CUDA execution (T03 / M1)

Candidate source compiles and executes only through `IsolatedGPUBackend`. Its public
operations are typed `prepare`, `build`, `run`, `run_sanitizer`, `cleanup`, plus fixed
benign probes. There is no candidate shell/flags interface or host fallback. Existing
`LocalBackend`/`TrustedLocalBackend` exact-source-hash authorization remains unchanged.
Containers share the host kernel and GPU driver; this is not isolation for actively
hostile native code. Such a threat model requires a VM or dedicated machine.

## Locked toolchain

`containers/toolchain.lock.json` records the actual pulled official CUDA 12.8.1 devel
RepoDigest and inspected local derived image ID. The backend invokes the immutable
derived ID, never the mutable tag. The derived image contains a trusted standard-library
Python runner; the controller and tests use the dedicated Python 3.11 environment.
Ubuntu's image helper interpreter is distribution Python 3.10 and needs no project
Python dependencies. CUDA supports the fixed `sm_89` target. CUDA compiler 12.8.93 and
Compute Sanitizer 2025.1.0.0 were observed in the image.

Rebuild commands (image-build networking is permitted):

```bash
docker pull nvidia/cuda:12.8.1-devel-ubuntu22.04
docker image inspect nvidia/cuda:12.8.1-devel-ubuntu22.04 --format '{{json .RepoDigests}}'
docker build --build-arg CUDA_BASE=nvidia/cuda@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97 -t gpu-agent-cuda:t03 containers
docker image inspect gpu-agent-cuda:t03 --format '{{.Id}}'
sha256sum containers/Dockerfile containers/runner.py
```

Update lock fields only from the new inspect and hash output. The local image ID is
machine-local and not a registry RepoDigest: another machine must build/inspect and
record its own ID, or load that exact image. Apt package resolution during image build
is not reproducible from the Dockerfile alone; the inspected derived image ID pins the
actual installed dependency bytes. No placeholder digest is accepted.

## Fixed execution policy

Every operation uses UID/GID 65532:65532, `--network none`, `--cap-drop ALL`,
`--security-opt no-new-privileges`, `--read-only`, 4 CPUs, 4 GiB memory with no extra
swap, 64 PIDs, exactly GPU device 0, no Docker logging driver, and disabled core dumps.
`/tmp` is a `nosuid,nodev` tmpfs: 1 GiB for build, 256 MiB for run/sanitizer/probes.
`/dev/shm` is limited to 16 MiB. Build wall time is 120 seconds; run/sanitizer requests
default to 30 seconds and cannot exceed 300 seconds. Logs are limited to 2 MiB for
combined program stdout/stderr, plus 2 MiB for the separate sanitizer log. Input is
limited to 32 MiB, each source to 4 MiB, and binary exports to 64 MiB.

Only four explicitly hashed public source files are copied into a unique task directory.
The Docker bind is that directory at `/input`, read-only. No repository root, `.git`,
private/evaluator directory, home, credentials, Docker socket, or unrelated path is
mounted. Compilation writes only container tmpfs. A bounded JSON/base64 envelope exports
regular binary bytes; the controller validates type, encoding and size, then writes
an immutable source/binary snapshot. It never extracts a tarball or follows exported
symlinks. Larger transport allowance accommodates a bounded encoded binary; partial
transport output is discarded on interruption and never published as program output.

Each container has a random operation name and operation/owner labels. After a timeout
or cancellation the controller inspects the exact name and checks its operation label,
then explicitly stops and removes that container. It does not rely on killing the
Docker CLI or perform broad container cleanup. Failure to confirm removal prevents
success. `cleanup` only removes the exact registered task workspace; persistent evidence
remains in RunStore.

## Memcheck evidence

The fixed command inside the container is:

```bash
compute-sanitizer --tool memcheck --error-exitcode 86 --log-file /tmp/memcheck.log /input/vector_add
```

Program stdout, program stderr, and raw memcheck log are separate artifacts. Parsing
uses only the sanitizer channel. Findings take precedence over exit code; CLEAN
requires a final complete zero-error summary, exit zero, and no truncation, timeout,
cancellation or tool failure. The parser records source location, raw artifact reference,
completion status and parser version. Other sanitizer tools return typed UNSUPPORTED
until T07. Synthetic logs in `tests/unit/test_sanitizer.py` are unit inputs only and
must never be copied into benchmark evidence. Format references:
[NVIDIA Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

`EvidenceRepository.public_view(run_id)` loads observed §8.3 fields from RunStore,
validates every nested artifact's registration/hash/visibility and matching run ID,
and rejects sibling/evaluator references. It never reads paths found in sanitizer text.
This API scopes artifact reads for the supplied run; role authorization must bind the
run ID before exposing a view to an agent. Source paths in findings are observations,
not file-access authority.

## Live acceptance and current blocker

`case_0001` intentionally allocates exactly 257 float elements, launches 256-thread
blocks, and omits the index guard. Ordinary execution may succeed or fail; memcheck
must record the invalid global access with actual raw evidence. `case_0000` must remain
CLEAN under the same configuration and produce the expected sums.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest tests/integration/test_isolation.py --require-live -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/you/conda_env/agentic-gpu-debugger/bin/python -m pytest tests/gpu/test_oob.py --require-live -q
```

Ordinary tests skip when image/runtime/GPU support is unavailable. `--require-live`
turns those skips into failures. Skipped tests and CPU-only benign probes cannot satisfy
M1 acceptance. As observed on 2026-09-15: Ubuntu 22.04.5 x86_64, Docker 29.1.3,
only `runc`/`io.containerd.runc.v2` runtimes, no NVIDIA Container Toolkit, and no running
containers before work. Docker's GPU request fails. M1 remains blocked until runtime
installation and both live acceptance commands succeed with real GPU evidence.

## Operator-only NVIDIA runtime setup

No host packages/configuration were changed by T03. `sudo` needs an operator password.
Use the [official NVIDIA installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
to configure the apt keyring/repository and install the reviewed package version. The
guide inspected on 2026-09-15 specifies:

```bash
NVIDIA_CONTAINER_TOOLKIT_VERSION=1.20.0-1
sudo apt-get install -y \
  nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
  libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
docker ps --format '{{.ID}} {{.Names}} {{.Status}}'
sudo cp -a /etc/docker/daemon.json /etc/docker/daemon.json.pre-nvidia-t03
sudo nvidia-ctk runtime configure --runtime=docker
sudo diff -u /etc/docker/daemon.json.pre-nvidia-t03 /etc/docker/daemon.json
docker ps --format '{{.ID}} {{.Names}} {{.Status}}'
sudo systemctl restart docker
docker info --format '{{json .Runtimes}}'
```

Before using the backup command, choose a fresh backup filename if that exact file
already exists. Preserve the existing `registry-mirrors` and `dns` keys. Review the
configuration diff and coordinate downtime if either container listing has workloads;
do not restart while they are in use. Never use `--privileged`, manually pass GPU
device nodes, mount host drivers, or switch candidates to host execution to make tests pass.
