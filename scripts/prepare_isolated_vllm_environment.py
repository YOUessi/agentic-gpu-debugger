"""Restore the already installed vLLM stack to a separate private venv.

Run this inventory/copy utility with the OLD serving interpreter, with its
normal package search path (to recover the existing effective versions).
The NEW serving interpreter is always launched with -I and no user site.
No package is installed into ~/.local, no version is upgraded, and no
network download is needed. This is a relocation of installed distribution
files, not an assertion about an unrecorded historical pip freeze.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata as md
import json
import os
import pathlib
import shutil
import subprocess
import sys
import re
from urllib.parse import urlsplit

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOTS = [
    "vllm", "pip", "setuptools", "wheel", "typing_extensions", "torch",
    "torchvision", "torchaudio", "triton", "transformers", "numpy", "pydantic",
]
APT = {"markupsafe": "python3-markupsafe", "wheel": "python3-wheel",
       "psutil": "python3-psutil"}


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def distributions() -> dict:
    stack = list(ROOTS)
    result = {}
    while stack:
        name = stack.pop()
        key = canonicalize_name(name)
        if key in result:
            continue
        dist = md.distribution(name)
        result[key] = dist
        for raw in dist.requires or ():
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate({**default_environment(), "extra": ""}):
                continue
            assert req.specifier.contains(md.version(req.name), prereleases=True), raw
            stack.append(req.name)
    return result


def install_file(src, dst, expected_hash=None):
    if src.is_symlink() or not src.is_file():
        raise ValueError(f"not a regular package file: {src.name}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    actual = sha(src)
    if expected_hash is not None:
        mode, value = expected_hash.mode, expected_hash.value
        assert mode == "sha256"
        if base64.urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip("=") != value:
            raise ValueError(f"installed package RECORD hash mismatch: {src.name}")
    if dst.exists():
        assert dst.is_file() and not dst.is_symlink() and sha(dst) == actual
    else:
        shutil.copy2(src, dst)
        assert sha(dst) == actual
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--resume-relocation", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    target = args.target.absolute()
    assert not args.report.exists()
    assert not target.exists() or args.resume_relocation
    assert sys.version_info[:2] == (3, 10)
    assert pathlib.Path(sys.prefix).name == "llm-opt-serving"
    dists = distributions()
    assert dists["vllm"].version == "0.19.1"
    assert dists["torch"].version == "2.10.0"
    assert dists["typing-extensions"].version == "4.15.0"
    if target.exists():
        cfg = (target / "pyvenv.cfg").read_text()
        assert not target.is_symlink() and "include-system-site-packages = false" in cfg
        assert (target / "bin/python").is_file()
    else:
        subprocess.run([sys.executable, "-I", "-m", "venv", "--copies", "--without-pip",
                        str(target)], check=True)
    dest = target / "lib/python3.10/site-packages"
    expected = {}
    records = []
    record_collisions = []
    for name, dist in sorted(dists.items()):
        root = pathlib.Path(dist.locate_file("")).resolve(strict=True)
        roots = (
            pathlib.Path(sys.prefix, "lib/python3.10/site-packages").resolve(),
            pathlib.Path.home() / ".local/lib/python3.10/site-packages",
            pathlib.Path("/usr/lib/python3/dist-packages"),
        )
        assert root in roots, root
        installed = []
        if dist.files:
            files = [(root / str(f), pathlib.Path(str(f)), f.hash) for f in dist.files
                     if ".." not in f.parts and "__pycache__" not in f.parts
                     and not str(f).endswith(".pyc")]
        else:
            # Ubuntu's three distro-owned Python distributions have egg-info,
            # not wheel RECORD. Use the installed package inventory only.
            package = APT[name]
            entries = subprocess.check_output(["dpkg", "-L", package], text=True).splitlines()
            files = []
            for entry in entries:
                p = pathlib.Path(entry)
                if p.is_relative_to(root) and p.is_file() and not p.is_symlink():
                    files.append((p, p.relative_to(root), None))
        for src, rel, h in sorted(files, key=lambda x: str(x[1])):
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("package escaped target site-packages")
            # The original installation has one verified shared-path collision.
            # Preserve the existing effective bytes, not the overwritten copy
            # described by torch-c-dlpack-ext's old wheel RECORD.
            if name == "torch-c-dlpack-ext" and str(rel) == "build_backend.py":
                owner = dists["flashinfer-python"]
                other = next(f for f in owner.files if str(f) == str(rel))
                assert pathlib.Path(owner.locate_file(other)).resolve() == src.resolve()
                assert sha(src) == "d034bb09167e69d48f61ffbddf3a0f2e4f6b4233e76f295eaa11b38ac535b657"
                assert h is not None and other.hash is not None
                record_collisions.append({
                    "path": str(rel), "overwritten_record_owner": name,
                    "matching_record_owner": "flashinfer-python",
                    "effective_sha256": sha(src), "original_expected_hash": h.value,
                    "matching_record_hash": other.hash.value,
                })
                h = other.hash
            value = install_file(src, dest / rel, h)
            installed.append([str(rel), value])
        assert installed
        expected[name] = dist.version
        records.append({"name": name, "version": dist.version,
                        "source_package_root": str(root), "installed_file_count": len(installed),
                        "file_inventory_sha256": hashlib.sha256(
                            json.dumps(installed, separators=(",", ":")).encode()).hexdigest()})
        print("RELOCATED", name, dist.version, len(installed), flush=True)
    # Only the new environment is used from here onward.
    env = dict(os.environ)
    for k in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(k, None)
    env["PYTHONNOUSERSITE"] = "1"
    python = target / "bin/python"
    check = r"""
import importlib.metadata as m, json, pathlib, sys, torch, vllm, typing_extensions
prefix=pathlib.Path(sys.prefix).resolve()
assert pathlib.Path(sys.executable).parent.parent.resolve()==prefix
for mod in (torch,vllm,typing_extensions):
    assert pathlib.Path(mod.__file__).resolve().is_relative_to(prefix)
assert torch.__version__.split('+')[0]=='2.10.0'
assert torch.version.cuda=='12.8'
assert vllm.__version__=='0.19.1'
print(json.dumps({'executable':sys.executable,'prefix':str(prefix),
                  'torch':{'path':torch.__file__,'version':torch.__version__,'cuda':torch.version.cuda},
                  'vllm':{'path':vllm.__file__,'version':vllm.__version__},
                  'typing_extensions':{'path':typing_extensions.__file__,
                                       'version':m.version('typing_extensions')},
                  'installed':{x.metadata['Name']:x.version for x in m.distributions()},
                  'sys_path':sys.path}))
"""
    out = subprocess.check_output([str(python), "-I", "-c", check], env=env, text=True)
    identity = json.loads(out.splitlines()[-1])
    actual = {canonicalize_name(k):v for k,v in identity["installed"].items()}
    assert expected == actual, (set(expected)-set(actual),set(actual)-set(expected))
    freeze = subprocess.check_output([str(python), "-I", "-m", "pip", "freeze", "--all"],
                                     env=env, text=True)
    # The report stays on the operator's machine. Require standard pinned
    # package lines, rather than accidentally storing a credential-bearing URL.
    for line in freeze.splitlines():
        if " @ " not in line:
            if "://" in line:
                raise ValueError("unrecognized pip freeze entry requires manual review")
            continue
        name, origin = line.split(" @ ", 1)
        url = urlsplit(origin)
        # The existing vLLM installation came from a local wheel. Keep the
        # actual pip freeze on the authorized host, not in published reports.
        if not (name.lower() == "vllm" and url.scheme == "file"
                and url.hostname in (None, "localhost")
                and not url.username and not url.password and not url.query
                and (not url.fragment or re.fullmatch(r"sha256=[0-9a-f]{64}", url.fragment))):
            raise ValueError("unreviewed direct-origin entry in pip freeze")
    subprocess.run([str(python), "-I", "-m", "pip", "check"], env=env, check=True)
    report = {"schema_version":"isolated-serving-relocation-v1",
              "source_python":sys.executable, "target_python":str(python),
              "relocation_method":"copy exact installed package files into separate no-system-site venv; verify wheel RECORD and copied bytes",
              "package_records":records,"shared_path_record_collisions":record_collisions,
              "module_identity":identity,"pip_freeze":freeze,
              "pip_freeze_sha256":hashlib.sha256(freeze.encode()).hexdigest()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print("ISOLATED_ENVIRONMENT_READY", json.dumps({
        "python":str(python), "packages":len(records), "torch":identity["torch"],
        "vllm":identity["vllm"], "typing_extensions":identity["typing_extensions"],
        "pip_freeze_sha256":report["pip_freeze_sha256"], "report_sha256":sha(args.report)
    }), flush=True)


if __name__ == "__main__":
    main()
