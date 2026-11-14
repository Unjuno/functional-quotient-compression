# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Verify original A9 evidence contents from the external raw-data bundle."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[3]
    manifest=json.loads((Path(__file__).parent/'EVIDENCE.json').read_text())
    checked=[]
    for name,datum in manifest['files'].items():
        path=root/name
        if not path.exists(): raise FileNotFoundError(str(path))
        raw=path.read_bytes()
        if len(raw)!=datum['bytes'] or hashlib.sha256(raw).hexdigest()!=datum['sha256']:
            raise ValueError('sha/size mismatch: '+str(path))
        checked.append(name)
    for key,n in [('main',216),('replication',108)]:
        lock=json.loads((root/key/'train.lock.json').read_text())
        if lock['num_models']!=n:raise ValueError('wrong number of models')
        for name,rec in lock['models'].items():
            raw=(root/key/'models'/name).read_bytes()
            if len(raw)!=rec['bytes'] or hashlib.sha256(raw).hexdigest()!=rec['sha256']:
                raise ValueError('model artifact mismatch: '+key+'/'+name)
    print('A9_BUNDLE_VALID',len(checked),'manifest_files, 324 checkpoint hashes')

if __name__=='__main__':main()
