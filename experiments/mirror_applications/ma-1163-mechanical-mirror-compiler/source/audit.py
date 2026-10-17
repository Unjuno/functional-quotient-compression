"""Recompute fixed gates and file evidence from CSV. Does not tune any model."""
from __future__ import annotations
import json,hashlib,platform,os
from pathlib import Path
import pandas as pd
import torch, sklearn, scipy, numpy as np
from model import N,K,R,GAMMA

ROOT=Path(__file__).resolve().parent.parent
R=ROOT/'results'
def main():
    train=pd.read_csv(R/'baseline_development.csv')
    cmp=pd.read_csv(R/'compiled_fresh.csv')
    direct=pd.read_csv(R/'direct_training_fresh.csv')
    dev=pd.read_csv(R/'direction_dev_exploratory.csv')
    fresh=pd.read_csv(R/'direction_fresh_replication.csv')
    assert len(train)==3 and len(cmp)==30 and len(direct)==9 and len(dev)==48 and len(fresh)==36
    assert all(train.dev_accuracy >=.9)
    exact=cmp[(cmp.kind=='mirror')&(cmp.m==N)]
    assert len(exact)==3 and exact.weight_max_abs.max()<1e-9
    assert exact.test_logit_max_abs.max()<1e-8
    assert all(cmp.physical_trainable_scalars==cmp.m)
    assert not cmp.duplicated(['seed','kind','m']).any()
    assert not fresh.duplicated(['world','seed','m','mode']).any()
    mirror=fresh[(fresh.m==768)&(fresh['mode']=='mirror_top')]
    plain=fresh[(fresh.m==768)&(fresh['mode']=='plain_top')]
    world_stats={}
    for world,d in mirror.groupby('world'):
        drop=float(d.accuracy_drop_pp.mean())
        wp=plain[plain.world==world]
        mg=float(d.reconstructed_test_accuracy.mean()-wp.reconstructed_test_accuracy.mean())
        world_stats[str(world)]={'mean_drop_pp':drop,'mirror_minus_plain_accuracy_pp':100*mg,
                                'retention_pass':bool(drop<=2),'mirror_specific_pass':bool(100*mg>=1)}
    assert len(world_stats)==3
    chosen=int(mirror.payload_bytes.iloc[0]); dense_bytes=int(cmp.teacher_payload_bytes.iloc[0])
    extra=(ROOT/'source/model.py').stat().st_size+(ROOT/'source/dev_direction_search.py').stat().st_size
    files=['PROTOCOL_FROZEN.json','PROTOCOL_EXPLORATORY_DIRECTION.json','PROTOCOL_FRESH_REPLICATION_FROZEN.json',
           'source/model.py','source/run.py','source/dev_direction_search.py','source/replicate_direction.py',
           'source/audit.py','tests/test_model.py','tests/test_compression.py',
           'results/baseline_development.csv','results/compiled_fresh.csv',
           'results/direct_training_fresh.csv','results/direction_dev_exploratory.csv',
           'results/direction_fresh_replication.csv']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    result={
      'experiment':'MA-1163-TRAINED-BASELINE-MIRROR-COMPILER','status':'SCREENING_MIRROR_SPECIFIC_FAIL',
      'native_dev_all_pass':True,
      'exact_m1210_weight_pass':True,'max_exact_weight_error':float(exact.weight_max_abs.max()),
      'max_exact_logit_error':float(exact.test_logit_max_abs.max()),
      'native_fresh_test_accuracy_mean':float(cmp[cmp.kind=='mirror'].groupby('seed').teacher_test_acc.first().mean()),
      'n_primary_rows':len(cmp),'n_direct_rows':len(direct),'n_dev_exploratory_rows':len(dev),'n_fresh_replication_rows':len(fresh),
      'replication_worlds':world_stats,'retention_worlds_passed':sum(x['retention_pass'] for x in world_stats.values()),
      'mirror_specific_worlds_passed':sum(x['mirror_specific_pass'] for x in world_stats.values()),
      'mean_fresh_native_accuracy':float(mirror.native_test_accuracy.mean()),
      'mean_fresh_mirror_m768_accuracy':float(mirror.reconstructed_test_accuracy.mean()),
      'mean_fresh_plain_m768_accuracy':float(plain.reconstructed_test_accuracy.mean()),
      'm768_mirror_payload_bytes_including_index':chosen,
      'native_inference_payload_bytes':dense_bytes,
      'decoder_python_source_bytes_model_plus_sparse_class':extra,
      'm768_with_decoder_code_bytes_conservative':chosen+extra,
      'm768_storage_payload_savings_ratio':float(1-chosen/dense_bytes),
      'm768_full_code_accounted_savings_ratio':float(1-(chosen+extra)/dense_bytes),
      'hardware':'AMD EPYC 9V74 virtual CPU, 5 vCPU, one torch thread',
      'clock':'not exposed/pinned; no cycle counter measurements',
      'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
      'numeric_format':'IEEE754 FP64 only; no arbitrary/high-bit precision',
      'class_data':'sklearn digits grayscale 8x8 images 1797, labels0..9',
      'files_sha256':hashes,
      'scope':['No true compression when executable source counted once per tiny model',
               'Selecting addresses from already fitted weights is a model-dependent code and index bytes are counted',
               'Fresh partitions share one 1797-image corpus; do not claim natural multi-domain transfer',
               'Mirror class is not superior to sparse plain orthogonal control']}
    path=ROOT/'VERIFICATION.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('files_sha256','scope')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
