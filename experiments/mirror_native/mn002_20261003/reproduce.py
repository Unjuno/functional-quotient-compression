# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Offline full reproduction: 24 MN002 + 9 MN002R training runs, locks, audits, files."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from train import run
from audit import freeze,audit_all
from verify_artifacts import export_and_verify
ROOT=Path(__file__).resolve().parent

def main(output: Path):
    if output.exists(): raise FileExistsError('Use a new output directory; never overwrite results.')
    output.mkdir(parents=True)
    for filename,tag in [('protocol.json','main'),('followup_protocol.json','followup')]:
        pp=ROOT/filename;p=json.loads(pp.read_text());root=output/tag;root.mkdir()
        for seed in p['seeds']:
            for variant in p['variants']:run(variant['name'],seed,root/'train',pp)
        freeze(root/'train',pp,root/'FROZEN_CHECKPOINTS.json')
        audit_all(root/'train',pp,root/'FROZEN_CHECKPOINTS.json',root/'audit')
        export_and_verify(root/'train',pp,root/'weights')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    main(ap.parse_args().output)
