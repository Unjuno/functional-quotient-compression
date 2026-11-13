#!/usr/bin/env python3
import csv, io, math, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'source/results'; S=4; P=256; A=6
DEV=[90301,90307]; FRESH=[90331,90333,90339]
def so3(v):
    a=np.linalg.norm(v); I=np.eye(3)
    if a<1e-9:return I
    k=v/a; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return I+math.sin(a)*K+(1-math.cos(a))*(K@K)
def world(seed):
    rng=np.random.default_rng(seed)
    clouds=[]
    for _ in range(S):
        pts=rng.normal(size=(P,3)).astype(np.float32); pts/=np.maximum(np.linalg.norm(pts,axis=1,keepdims=True),1e-6); pts*=rng.uniform(.3,1.2,(P,1)).astype(np.float32); clouds.append(pts)
    clouds=np.stack(clouds)
    acts=[]
    for _ in range(A):
        M=np.eye(4); M[:3,:3]=so3(rng.normal(0,.25,3)); M[:3,3]=rng.normal(0,.2,3); acts.append(M.astype(np.float32))
    return clouds,np.stack(acts)
def npz(arrays):
    b=io.BytesIO();np.savez_compressed(b,**arrays);return b.getvalue()
def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'payloads').mkdir(exist_ok=True);rows=[];manifest=[]
    for split,seeds in [('development',DEV),('fresh',FRESH)]:
      for seed in seeds:
        clouds,acts=world(seed); pairs=[(i,j) for i in range(A) for j in range(A)]
        logical=[]
        for s in range(S):
          for i,j in pairs:
            M=acts[j]@acts[i]; q=clouds[s]@M[:3,:3].T+M[:3,3]; logical.append(q)
        logical=np.stack(logical)
        common={'canonical':clouds,'actions':acts,'meta':np.frombuffer(b'SE3;S4;P256;A6;pair-composition',dtype=np.uint8)}
        mirror_blob=npz(common)
        native_blob=mirror_blob
        dense_blob=npz({'posed_points':logical,'meta':np.frombuffer(b'S4;P256;A36;XYZ-f32',dtype=np.uint8)})
        start=time.perf_counter()
        for name,blob in [('mirror_se3',mirror_blob),('native_se3',native_blob),('independent_posed_clouds',dense_blob)]:
          path=OUT/'payloads'/f'{split}_{seed}_{name}.npz';path.write_bytes(blob)
          with np.load(path,allow_pickle=False) as z: loaded={k:z[k] for k in z.files}
          assert np.isfinite(list(loaded.values())[0]).all()
          if name=='mirror_se3':
            out=[]
            for s in range(S):
              for i,j in pairs:
                M=loaded['actions'][j]@loaded['actions'][i]; out.append(loaded['canonical'][s]@M[:3,:3].T+M[:3,3])
            replay=np.stack(out)
          elif name=='native_se3':
            out=[]
            for s in range(S):
              for i,j in pairs:
                M=loaded['actions'][j]@loaded['actions'][i]; out.append(loaded['canonical'][s]@M[:3,:3].T+M[:3,3])
            replay=np.stack(out)
          else: replay=loaded['posed_points']
          delta=np.abs(replay-logical);errmax=float(delta.max());rmse=float(np.sqrt(np.mean(delta**2)))
          assert errmax<1e-5
          if name!='independent_posed_clouds':
            mats=loaded['actions']
            rot=mats[:,:3,:3]
            assert float(np.max(np.abs(rot@np.transpose(rot,(0,2,1))-np.eye(3))))<2e-5
            assert float(np.min(np.linalg.det(rot)))>.999
          mac= S*A*A*(9+3) if name!='independent_posed_clouds' else 0
          rows.append({'experiment_id':'MA-903','status':'FAIL','condition':f'{split}:{name}','seed':seed,'quality':f'max_error={errmax:.9g};rmse={rmse:.9g}','actual_payload_bytes':len(blob),'compute_proxy':mac,'wall_clock_seconds':f'{time.perf_counter()-start:.6f}','notes':'Actual NPZ payload roundtrip; synthetic rigid geometry only.'})
        manifest.append({'split':split,'seed':seed,'objects':S,'points_per_object':P,'actions':A,'heldout_compositions':len(pairs),'fresh':split=='fresh'})
    with (OUT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (OUT/'split_manifest.json').write_text(__import__('json').dumps(manifest,indent=2)+'\n')
    print('rows',len(rows),'fresh',FRESH)
if __name__=='__main__':main()
