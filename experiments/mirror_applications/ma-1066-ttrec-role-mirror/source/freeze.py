#!/usr/bin/env python3
"""Freeze MA-1066 protocol, runner, selection, and vendored TT backend before audit."""
import hashlib, json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
if not torch.cuda.is_available(): raise SystemExit('Freeze requires the reviewed CUDA runtime; do not create a CPU audit freeze')
required=[ROOT/'PROTOCOL.json',ROOT/'source'/'train.py',ROOT/'runs'/'DEVELOPMENT_SELECTION.json',ROOT/'source'/'vendor'/'FBTT-Embedding'/'tt_embeddings_ops.py',ROOT/'source'/'vendor'/'FBTT-Embedding'/'tt_embeddings_cuda.cu']
missing=[str(p) for p in required if not p.is_file()]
if missing: raise SystemExit('Cannot freeze; missing: '+', '.join(missing))
freeze={'experiment_id':'MA-1066','protocol_sha256':sha(required[0]),'runner_sha256':sha(required[1]),'selection_sha256':sha(required[2]),'tt_ops_sha256':sha(required[3]),'tt_cuda_sha256':sha(required[4]),'upstream_commit':'b95947ccd32cf9ab851242cc606562f2a74d9efa','torch_version':torch.__version__,'cuda_version':torch.version.cuda,'gpu_model':torch.cuda.get_device_name(0),'audit_opened':False,'cache_policy':'disabled'}
out=ROOT/'FRESH_FREEZE.json'
if out.exists(): raise SystemExit('FRESH_FREEZE.json already exists; do not overwrite an audit freeze')
out.write_text(json.dumps(freeze,indent=2)+'\n')
print(json.dumps(freeze,indent=2))
