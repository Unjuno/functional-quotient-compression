#!/usr/bin/env python3
"""Independent input/serialization/metric verification for MA-1053 saved results."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'runs/raw'; RUNS=ROOT/'runs'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b,tol=1e-8):
    if abs(float(a)-float(b))>tol: raise AssertionError(f'{a} != {b} (tol={tol})')
def class_metrics(y,p):
    pred=np.asarray(p).argmax(-1); y=np.asarray(y); f=[]
    for c in range(5):
        tp=np.sum((pred==c)&(y==c)); fp=np.sum((pred==c)&(y!=c)); fn=np.sum((pred!=c)&(y==c)); f.append(2*tp/(2*tp+fp+fn+1e-12))
    return float(np.mean(pred==y)),float(np.mean(f))
def main():
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text()); freeze=json.loads((RUNS/'FRESH_FREEZE.json').read_text())
    assert sha(ROOT/freeze['frozen_protocol_snapshot'])==freeze['frozen_protocol']['sha256']
    assert sha(ROOT/freeze['frozen_runner_snapshot'])==freeze['frozen_runner']['sha256']
    manifest=json.loads((RUNS/'input_manifest.json').read_text())
    for name,entry in manifest['files'].items():
        path=(RAW/name) if '/' not in name else Path('/home/agent/.cache/huggingface/hub/models--AutonLab--MOMENT-1-base/snapshots/5e44b0ea26376a176360f87831124e018f876d96')/name.split('/',1)[1]
        assert path.stat().st_size==entry['size_bytes'] and sha(path)==entry['sha256'],name
    dev=json.loads((RUNS/'dev/run.json').read_text()); aud=json.loads((RUNS/'audit/run.json').read_text())
    payloads=json.loads((RUNS/'payload_hashes.json').read_text())['payloads']
    for split,methods in payloads.items():
        for method,components in methods.items():
            for filename,entry in components.items():
                path=RUNS/split/filename
                assert path.stat().st_size==entry['size_bytes'] and sha(path)==entry['sha256'],(split,filename)
    assert aud['seed']==105302 and aud['split']=='audit' and dev['split']=='dev'
    assert aud['native_prehead_parity_max_abs_delta']==0.0
    for name,v in aud['methods'].items():
        assert v['serialized_replay_max_delta']==0.0
        weight=RUNS/'audit'/f'{name}.safetensors'; meta=weight.with_suffix('.json')
        payload_meta=json.loads(meta.read_text()); active=name in ('diag','mirror')
        assert payload_meta['task_codes_per_role']==(8 if active else 0)
        assert payload_meta['task_codes_total']==(24 if active else 0)
        assert weight.stat().st_size+meta.stat().st_size==v['adaptation_bytes']
        assert v['full_payload_bytes']==aud['common_base_and_config_bytes']+v['adaptation_bytes']
    for t,v in aud['native'].items():
        if t!='anomaly':
            weight=RUNS/'audit'/f'native_{t}.safetensors'
            assert weight.stat().st_size+weight.with_suffix('.json').stat().st_size==v['adaptation_bytes']
            assert v['serialized_replay_max_delta']==0.0
    # Recompute audit forecast scores from raw ETTm2 and saved forecasts.
    df=pd.read_csv(RAW/'ETTm2.csv'); vals=df['OT'].to_numpy(np.float32); ntr=34560; nv=11520; nte=11520
    z=(vals-float(vals[:ntr].mean()))/float(vals[:ntr].std()); starts=np.arange(ntr+nv,ntr+nv+nte-96+1,96,dtype=int)
    y=np.stack([z[s+512:s+512+96] for s in starts]); xm=np.asarray([z[s:s+512].mean() for s in starts])[:,None]; xs=np.asarray([z[s:s+512].std()+1e-5 for s in starts])[:,None]
    scale=float(np.mean(np.abs(z[96:ntr]-z[:ntr-96])))
    for name,v in aud['methods'].items():
        p=np.asarray(v['predictions']['forecast']); p=p[:,0,:] if p.ndim==3 else p; p=p*xs+xm; e=p-y
        close(np.mean(np.abs(e))/scale,v['metrics']['forecast']['mase'])
        close(np.mean(e**2),v['metrics']['forecast']['mse'])
    # Recompute classification from official ECG5000 test class labels.
    labels=[]
    for line in (RAW/'ECG5000_TEST.ts').read_text().splitlines():
        if line and not line.startswith('@') and not line.startswith('#'):
            labels.append(int(line.rsplit(':',1)[1]))
    train_labels=[]
    for line in (RAW/'ECG5000_TRAIN.ts').read_text().splitlines():
        if line and not line.startswith('@') and not line.startswith('#'):
            train_labels.append(int(line.rsplit(':',1)[1]))
    classes=sorted(set(train_labels)); mapping={v:i for i,v in enumerate(classes)}; yy=np.asarray([mapping[v] for v in labels])
    for name,v in aud['methods'].items():
        acc,f1=class_metrics(yy,v['predictions']['classification']); close(acc,v['metrics']['classification']['accuracy']); close(f1,v['metrics']['classification']['macro_f1'])
    # Recompute anomaly point metrics on non-overlapping suffix windows.
    arr=pd.read_csv(RAW/'198_UCR_Anomaly_tiltAPB2_50000_124159_124985.out',header=None).to_numpy(np.float32); values=arr[:,0]; labs=arr[:,1].astype(int)
    starts=np.arange(50000,len(values),512,dtype=int); targets=[]; ylab=[]; mus=[]; sds=[]
    for st in starts:
        raw=values[st:min(st+512,len(values))]; targets.append(raw); ylab.append(labs[st:st+len(raw)]); mus.append(float(raw.mean())); sds.append(float(raw.std()+1e-6))
    for name,v in aud['methods'].items():
        pred=np.asarray(v['predictions']['anomaly']); scores=[]
        for i,raw in enumerate(targets):
            rec=pred[i,0].reshape(-1)[:len(raw)]*sds[i]+mus[i]; scores.extend(((rec-raw)**2).tolist())
        scores=np.asarray(scores); labels=np.concatenate(ylab).astype(bool); threshold=dev['methods'][name]['metrics']['anomaly']['calibration_error_q99']; positive=scores>threshold
        tp=int(np.sum(positive&labels)); fp=int(np.sum(positive&~labels)); fn=int(np.sum(~positive&labels)); prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-12)
        close(f1,v['metrics']['anomaly']['point_f1']); assert tp==v['metrics']['anomaly']['tp'] and fp==v['metrics']['anomaly']['fp'] and fn==v['metrics']['anomaly']['fn']
        close(average_precision_score(labels.astype(int),scores),v['metrics']['anomaly']['auprc'],tol=1e-7)
    print('MA-1053 verification: input hashes, freeze hashes, native parity, saved-payload sizes/replay and audit metrics all pass.')
if __name__=='__main__': main()
