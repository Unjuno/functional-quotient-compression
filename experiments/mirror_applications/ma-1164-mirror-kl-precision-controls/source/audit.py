"""Post-fresh evidence audit. No hyperparameter selection from fresh metrics."""
from __future__ import annotations
import hashlib,json,sys,platform
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent
RES=ROOT/'results'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    freeze=json.loads((ROOT/'FREEZE.json').read_text())
    second=json.loads((ROOT/'FRESH_FREEZE.json').read_text())
    selected=json.loads((RES/'development_selection.json').read_text())
    assert second['development_selection_sha256']==sha(RES/'development_selection.json')
    assert second['protocol_sha256']==sha(ROOT/'PROTOCOL.json')
    assert second['core_sha256']==sha(ROOT/'source/core.py')
    assert second['source_sha256']==sha(ROOT/'source/run.py')
    for name,expected in freeze['files_sha256'].items():
        # STATUS.md is an intentionally mutable post-run status record; its original hash
        # remains in FREEZE.json. All code, protocol and preregistered README stay pinned.
        if name=='STATUS.md':continue
        assert sha(ROOT/name)==expected,(name,'frozen source modified')
    d=pd.read_csv(RES/'development_sweep.csv')
    f=pd.read_csv(RES/'fresh_results.csv')
    assert len(d)==118,len(d)
    assert len(f)==102,len(f)
    assert set(d.world)=={618410,618411}
    assert set(f.world)=={618420,618421,618422}
    assert set(f.seed)=={31,32}
    assert set(d.world).isdisjoint(set(f.world))
    assert f.groupby(['world','seed']).size().eq(17).all()
    assert not f[['world','seed','m','kind','k','rounds']].duplicated().any()
    assert f[(f.kind.str.startswith('native'))].groupby(['world','seed']).size().eq(3).all()
    main_k={int(k):(v['k'],v['r']) for k,v in selected['selected'].items()}
    worlds=[]
    for world in sorted(f.world.unique()):
        for m in (512,768):
            k,r=main_k[m]
            mirror=f[(f.world==world)&(f.m==m)&(f.kind=='mirror')&(f.k==k)&(f.rounds==r)]
            native=f[(f.world==world)&(f.kind=='native_float16')]
            plain=f[(f.world==world)&(f.m==m)&(f.kind=='plain')]
            raw=f[(f.world==world)&(f.m==m)&(f.kind=='raw')]
            svd=f[(f.world==world)&(f.m==m)&(f.kind=='svd_matched')]
            assert len(mirror)==len(native)==len(plain)==len(raw)==len(svd)==2
            byte_cap=int(mirror.serialized_bytes.iloc[0]); assert (mirror.serialized_bytes==byte_cap).all()
            best_nonmirror=max([float(v.accuracy.mean()) for v in (native,plain,raw,svd)])
            # Use same-world paired seed mean, never a shuffled comparison.
            acc=float(mirror.accuracy.mean());native_acc=float(native.accuracy.mean());plain_acc=float(plain.accuracy.mean())
            win=acc-best_nonmirror>=.01
            retention=native_acc-acc<=.02
            worlds.append({'world':int(world),'m':m,'k':k,'rounds':r,
                'mirror_accuracy':acc,'native_fp16_accuracy':native_acc,
                'plain_accuracy':plain_acc,'svd_accuracy':float(svd.accuracy.mean()),
                'accuracy_drop_pp':100*(native_acc-acc),'mirror_minus_plain_pp':100*(acc-plain_acc),
                'mirror_minus_best_nonmirror_pp':100*(acc-best_nonmirror),
                'retention_pass':bool(retention),'mirror_specific_pass':bool(win),'payload_cap_bytes':byte_cap})
    summary=(f.groupby(['m','kind','k','rounds'],dropna=False)
             .agg(accuracy=('accuracy','mean'),nll=('nll','mean'),payload_bytes=('serialized_bytes','mean'),
                  infer_batch64_ms=('infer_batch64_ms','mean'),weight_l2=('weight_l2','mean'))
             .reset_index())
    summary.to_csv(RES/'summary_by_model.csv',index=False)
    code_bytes=(ROOT/'source/core.py').stat().st_size
    freshrank=f[f.kind=='svd_matched'][['m','svd_rank','serialized_bytes']].drop_duplicates().to_dict('records')
    metadata={'id':'MA-1164','status':'SCREENING', 'overall_mirror_specific':'FAIL',
      'frozen_protocol_verified':True,'source_sha_verified':True,'dev_rows':len(d),'fresh_rows':len(f),
      'fresh_worlds':sorted(f.world.unique().tolist()),'fresh_seed_count':2,
      'selected_K_R':selected['selected'],'world_gates':worlds,
      'mirror_specific_pass_worlds_m768':sum(z['mirror_specific_pass'] for z in worlds if z['m']==768),
      'retention_pass_worlds_m768':sum(z['retention_pass'] for z in worlds if z['m']==768),
      'native_16bit_acc':float(f[f.kind=='native_float16'].accuracy.mean()),
      'native_16bit_payload_bytes':int(f[f.kind=='native_float16'].serialized_bytes.iloc[0]),
      'mirror_m768_acc':float(f[(f.kind=='mirror')&(f.m==768)&(f.k==main_k[768][0])&(f.rounds==main_k[768][1])].accuracy.mean()),
      'mirror_m768_payload_bytes':int(f[(f.kind=='mirror')&(f.m==768)].serialized_bytes.iloc[0]),
      'mirror_decoder_source_bytes':code_bytes,
      'mirror_m768_payload_plus_source_bytes':int(f[(f.kind=='mirror')&(f.m==768)].serialized_bytes.iloc[0])+code_bytes,
      'svd_ranks_bytes':freshrank,
      'hardware':json.loads((RES/'runtime_environment.json').read_text()),
      'pinned_sha256':freeze['files_sha256'],
      'files_sha256':{name:sha(RES/name) for name in ('development_sweep.csv','development_selection.json','fresh_results.csv')},
      'warnings':['Some fresh test images are reused in other worlds because each world draws a new split of the SAME fixed sklearn digits dataset. No independent-domain conclusion.',
                  'Fine-tuned model rows reuse pre-fine-tune weight_l2 and latency fields; only fine-tuned accuracy/NLL and elapsed seconds reflect adapted model.',
                  'FP16/FP32 stored weights were cast back to FP64 for quality evaluation, not true low-precision kernel latency.',
                  'Python decoder source-byte charge may be amortized across many deployments; report both unamortized and payload-only costs.',
                  'Inference wall medians affected by CPU clock and cache jitter; no hardware calibration. u_c not estimated.',
                  'K,R selected on development only. No fresh relabeling or retuning.']}
    (RES/'verification.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    print('AUDIT VERIFIED',len(f),'rows', 'M768 retention worlds',metadata['retention_pass_worlds_m768'],
           'Mirror specific',metadata['mirror_specific_pass_worlds_m768'])
    print('summary')
    print(f.groupby(['m','kind']).accuracy.mean().round(4).to_string())
    for item in worlds:print('world',item)

if __name__=='__main__':main()
