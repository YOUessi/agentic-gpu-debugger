#!/usr/bin/env python3
"""Opt-in, real pretrained CPU validation; no test doubles or network at inference.

This is a small controlled smoke test, not a retrieval quality benchmark.
Run inside an isolated venv containing torch==2.5.1+cpu,
sentence-transformers==3.4.1, and transformers==4.48.3. The optional runtime
site-packages path is appended read-only to reuse the project's locked deps.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
sys.dont_write_bytecode = True

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--runtime-site-packages', type=Path, action='append', default=[])
args = parser.parse_args()
for runtime_path in args.runtime_site_packages:
    sys.path.append(str(runtime_path.resolve()))
sys.path.insert(0, str(args.source.resolve() / 'src'))
os.environ['HF_HOME'] = str(args.output.parent / 'runtime-cache')

import torch
from gpu_agent.harness.graph_memory import (
    LocalSentenceEmbedder, SQLiteRepairStore, MEMORY_VERSION,
    make_lesson, rank_texts, sanitize_lesson, tokens,
)

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
network_attempts = []
def reject_network(self, address):
    network_attempts.append(str(address))
    raise AssertionError('Network is forbidden during local model validation')
socket.socket.connect = reject_network

start = time.monotonic()
result = {
    'test_kind': 'controlled real pretrained CPU embedding smoke validation',
    'source_commit': subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
    'graph_memory_sha256': hashlib.sha256((args.source/'src/gpu_agent/harness/graph_memory.py').read_bytes()).hexdigest(),
    'model_repository': 'sentence-transformers/paraphrase-MiniLM-L3-v2',
    'model_revision': '4ca70771034acceecb2e72475f72050fcdde4ddc',
    'model_weight_sha256': hashlib.sha256((args.model/'model.safetensors').read_bytes()).hexdigest(),
    'expected_weight_sha256_from_official_huggingface': 'cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433',
    'official_weight_source': 'https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/blob/main/model.safetensors',
    'versions': {p: importlib.metadata.version(p) for p in ['torch', 'sentence-transformers', 'transformers', 'langgraph']},
    'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
    'torch_cuda_build': torch.version.cuda,
    'functional_checks': {},
}
checks = result['functional_checks']
checks['official_weight_hash_matches'] = result['model_weight_sha256'] == result['expected_weight_sha256_from_official_huggingface']
checks['cpu_only_torch_build'] = torch.version.cuda is None
checks['model_files_are_regular_without_symlinks'] = all(not p.is_symlink() and (p.is_dir() or p.is_file()) for p in args.model.rglob('*'))
if not all(checks.values()):
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    raise SystemExit('Model integrity or CPU prerequisite failed')

embedder = LocalSentenceEmbedder(args.model)
result['embedding_identity'] = embedder.identity
result['model_device'] = str(embedder.model.device)
checks['all_model_parameters_on_cpu'] = all(p.device.type == 'cpu' for p in embedder.model.parameters())
probe = embedder.embed('A physician examines an ill child.')
repeat = embedder.embed('A physician examines an ill child.')
result['embedding_dimension'] = len(probe)
result['embedding_norm'] = math.sqrt(sum(x*x for x in probe))
result['repeat_max_abs_delta'] = max(abs(a-b) for a,b in zip(probe, repeat))
checks['real_384_dimension_finite_normalized_embedding'] = len(probe) == 384 and all(math.isfinite(x) for x in probe) and abs(result['embedding_norm']-1) < 1e-5
checks['repeated_encode_identical'] = probe == repeat

cases = [
    ('compile', 'A missing closing brace causes the compiler to reject the CUDA kernel.', 'Correct malformed punctuation during translation'),
    ('numeric', 'Subtract the largest logit before exponentiation to prevent overflow in softmax.', 'Normalize probabilities safely when exponential values explode'),
    ('ast', 'Refresh the source hash and rebind the AST node after editing a file.', 'Locate updated syntax tree targets following a revision'),
    ('infrastructure', 'The remote service is unavailable because its connection timed out.', 'Network outage blocks communication'),
    ('generic_paraphrase_control', 'The physician examines a sick child.', 'doctor treats ill youngster'),
]
ns = (MEMORY_VERSION, 'controlled_semantic_demo', 'repo_a')
other_ns = (MEMORY_VERSION, 'controlled_semantic_demo', 'repo_b')
with tempfile.TemporaryDirectory(prefix='real-embedding-', dir=args.output.parent) as temporary:
    database = Path(temporary)/'memory.sqlite3'
    writable = SQLiteRepairStore(database, ns, embedder=embedder)
    for key, text, _ in cases:
        writable.put(ns, key, {'query_text': text, 'fixture': 'controlled illustrative text'})
    other = SQLiteRepairStore(database, other_ns, embedder=embedder)
    other.put(other_ns, 'OTHER_SCOPE_SENTINEL', {'query_text': cases[0][2]})
    reader = SQLiteRepairStore(database, ns, embedder=embedder, read_only=True)
    lexical_reader = SQLiteRepairStore(database, ns, read_only=True)
    before_digest = hashlib.sha256(database.read_bytes()).hexdigest()
    before_stat = database.stat()
    retrieval = []
    for expected, text, query in cases:
        hits = reader.search(ns, query=query, limit=5)
        baseline = lexical_reader.search(ns, query=query, limit=5)
        bm25 = rank_texts([c[1] for c in cases], query)
        item = {
            'query': query, 'expected': expected, 'expected_document_text': text,
            'top1': hits[0].key if hits else None,
            'top1_correct': bool(hits) and hits[0].key == expected,
            'ranking': [{'key':h.key, 'score':h.score} for h in hits],
            'bm25_scores': bm25,
            'lexical_only_hits': [h.key for h in baseline],
            'semantic_expected_rank': next((i+1 for i,h in enumerate(hits) if h.key == expected), None),
            'bm25_expected_rank': next((i+1 for i,h in enumerate(baseline) if h.key == expected), None),
            'bm25_only_top1_correct': bool(baseline) and baseline[0].key == expected,
            'bm25_candidate_scores': [{'key':c[0], 'score':score, 'returned_rank':next((i+1 for i,h in enumerate(baseline) if h.key == c[0]), None)} for c,score in zip(cases,bm25)],
            'query_expected_document_token_overlap': sorted(set(tokens(query)) & set(tokens(text))),
        }
        retrieval.append(item)
    result['retrieval_cases'] = retrieval
    checks['all_queries_have_zero_lexical_score'] = all(not any(x['bm25_scores']) for x in retrieval)
    checks['lexical_only_returns_no_results'] = all(not x['lexical_only_hits'] for x in retrieval)
    result['retrieval_metrics'] = {
        'sample_size': len(retrieval),
        'semantic_top1_hits': sum(x['top1_correct'] for x in retrieval),
        'bm25_only_top1_hits': sum(x['bm25_only_top1_correct'] for x in retrieval),
        'semantic_top1_fraction': sum(x['top1_correct'] for x in retrieval)/len(retrieval),
        'bm25_only_top1_fraction': sum(x['bm25_only_top1_correct'] for x in retrieval)/len(retrieval),
        'semantic_perfect_recall': all(x['top1_correct'] for x in retrieval),
        'note': 'Fixed controlled smoke fixtures, not a held-out quality benchmark. Queries were not retuned after observing the compile-query miss.',
    }
    checks['namespace_sentinel_never_returned'] = all(h['key'] != 'OTHER_SCOPE_SENTINEL' for x in retrieval for h in x['ranking'])
    checks['list_namespaces_is_bound_scope_only'] = reader.list_namespaces() == [ns]
    refused = {}
    for name, operation in {
        'read_only_put': lambda: reader.put(ns, 'unexpected', {'query_text':'forbidden'}),
        'read_only_delete': lambda: reader.delete(ns, 'compile'),
        'wrong_scope_get': lambda: reader.get(other_ns, 'OTHER_SCOPE_SENTINEL'),
        'wrong_scope_search': lambda: reader.search(other_ns, query=cases[0][2]),
        'wrong_scope_put': lambda: writable.put(other_ns, 'unexpected', {'query_text':'forbidden'}),
    }.items():
        try:
            operation()
            refused[name] = False
        except ValueError as exc:
            refused[name] = str(exc)
    result['rejected_operations'] = refused
    checks['read_only_and_scope_violations_rejected'] = all(refused.values())
    after_stat = database.stat()
    checks['readonly_database_bytes_and_mtime_unchanged'] = hashlib.sha256(database.read_bytes()).hexdigest() == before_digest and after_stat.st_mtime_ns == before_stat.st_mtime_ns
    reopened_embedder = LocalSentenceEmbedder(args.model)
    reopened_reader = SQLiteRepairStore(database, ns, embedder=reopened_embedder, read_only=True)
    rerun = [[(h.key,h.score) for h in reopened_reader.search(ns, query=q, limit=5)] for _,_,q in cases]
    previous = [[(h['key'],h['score']) for h in x['ranking']] for x in retrieval]
    checks['reopened_model_identity_identical'] = reopened_embedder.identity == embedder.identity
    checks['reopened_store_rankings_and_scores_identical'] = rerun == previous
    with sqlite3.connect(database) as connection:
        rows = connection.execute('SELECT embedding,embedding_id FROM lessons').fetchall()
    checks['sqlite_persisted_real_384_dimension_vectors'] = len(rows) == 6 and all(len(json.loads(v)) == 384 and identity == embedder.identity for v,identity in rows)
    result['stored_embedding_rows'] = len(rows)

state = {
    'identity': {'run_id':'a'*32}, 'effects_head_sha256':'b'*64, 'turns':2,
    'snapshot':{}, 'finish_accepted':False,
    'evidence':[
        {'turn':1,'tool':'repo_compile_candidate','status':'FAILED'},
        {'turn':2,'tool':'repo_compile_candidate','status':'OK'},
    ],
    'events':[{'kind':'failure_routed','recovery':{'strategy':'COMPILE_REPAIR','failure':{'turn':1}}}],
}
lesson = make_lesson(state)
assert lesson is not None
canary = 'UNTRUSTED_RAW_SOURCE_OR_PROMPT_CANARY'
lesson.update({'query_text':canary, 'kernel_source':canary, 'raw_model_response':canary})
lesson['strategies'].append(canary)
lesson['tool_sequence'].append(canary)
sanitized = sanitize_lesson(lesson)
checks['sanitized_lesson_removes_untrusted_text'] = sanitized is not None and canary not in json.dumps(sanitized) and sanitized['strategies'] == ['COMPILE_REPAIR']
checks['sanitized_lesson_retains_observed_recovery'] = sanitized['episodes'] == [{'strategy':'COMPILE_REPAIR','after_failure':[{'tool':'repo_compile_candidate','status':'OK'}]}]
result['sanitized_lesson'] = sanitized
result['network_attempts_during_inference'] = network_attempts
checks['offline_inference_no_network_attempts'] = not network_attempts
result['elapsed_seconds'] = round(time.monotonic()-start,3)
result['functional_passed_checks'] = sum(checks.values())
result['functional_total_checks'] = len(checks)
result['functional_all_passed'] = all(checks.values())
result['limitations'] = ['Controlled smoke fixtures only; no benchmark quality claim.', 'No GPU or paid provider inference.', 'Model transported from hf-mirror.com after direct HuggingFace access failed; safetensors SHA256 independently matches the official HuggingFace page.']
args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
raise SystemExit(0 if result['functional_all_passed'] else 1)
