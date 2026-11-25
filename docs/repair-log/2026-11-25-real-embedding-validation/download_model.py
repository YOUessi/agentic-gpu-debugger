#!/usr/bin/env python3
"""Fetch only content-locked model data; never Python code or pickle weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lock', type=Path, default=Path(__file__).parent/'results/model-lock.json')
parser.add_argument('--destination', type=Path, default=Path(__file__).parent/'model')
parser.add_argument('--endpoint', choices=['https://huggingface.co','https://hf-mirror.com'], default='https://huggingface.co')
args = parser.parse_args()
lock = json.loads(args.lock.read_text())
repo, revision = lock['model_repository'], lock['model_revision']
args.destination.mkdir(parents=True, exist_ok=True)
assert not args.destination.is_symlink()


def fetch(entry):
    name = entry['path']
    relative = Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    assert relative.suffix in {'.json', '.txt', '.md', '.safetensors'}
    path = args.destination/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.is_symlink()
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']:
        return {'path':name, 'sha256':entry['sha256'], 'bytes':path.stat().st_size, 'transport':'verified-existing-file'}
    routes = [f'resolve/{revision}']
    if relative.suffix != '.safetensors':
        routes.append('raw/main')
    for route in routes:
        url = f'{args.endpoint}/{repo}/{route}/{name}'
        request = urllib.request.Request(url, headers={'User-Agent':'huggingface_hub/0.36.0; python/3.11.0'})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if route == 'raw/main':
                    assert response.headers.get('X-Repo-Commit') == revision, 'Revision drift rejected'
                data = response.read()
            assert len(data) == entry['bytes'], 'Model file size mismatch'
            assert hashlib.sha256(data).hexdigest() == entry['sha256'], 'Model content mismatch'
            path.write_bytes(data)
            return {'path':name, 'sha256':entry['sha256'], 'bytes':len(data), 'transport':url}
        except Exception as exc:
            print(f'{name}: {type(exc).__name__}: {exc}', flush=True)
    raise RuntimeError(f'Unable to retrieve pinned content for {name}')

with ThreadPoolExecutor(max_workers=4) as pool:
    files = list(pool.map(fetch, lock['files']))
print(json.dumps({'model_repository':repo, 'model_revision':revision, 'files':files}, indent=2))
