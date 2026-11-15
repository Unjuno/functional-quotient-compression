"""Post-audit descriptive replay for preregistered distance-error metric."""
import importlib.util, json
from pathlib import Path
import numpy as np, faiss
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma1036',Path(__file__).with_name('run_experiment.py'))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
angles=json.loads((ROOT/'runs'/'DEV_SELECTION.json').read_text())['frozen_angles_rad']
rows=[]
for seed in (103601,103602,103603):
    _,db,qs=m.world(seed)
    for method in ('unrotated','shared','mirror','independent'):
        for j,name in enumerate(m.DOMAINS):
            p=ROOT/'payload'/method/str(seed)
            idxno=0 if method in ('unrotated','shared') else j
            idx=faiss.read_index(str(p/f'index_{idxno}.faiss'))
            q=qs[j]
            if method in ('shared','mirror','independent'):
                bno=j if method=='independent' else 0
                R=np.fromfile(p/f'basis_{bno}.f32',dtype='<f4').reshape(m.D,m.D)
                theta=angles[j] if method=='mirror' else 0.
                q=m.apply_basis(m.givens(q,theta),R)
                dv=m.apply_basis(m.givens(db,theta),R)
            else: dv=db
            d_hat,ids=idx.search(q,m.K)
            d_exact=np.sum((q[:,None,:]-dv[ids])**2,axis=2)
            err=d_hat-d_exact
            rows.append({'seed':seed,'domain':name,'method':method,'mae_squared_distance':float(np.mean(np.abs(err))), 'signed_squared_distance_error':float(np.mean(err))})
out={'note':'Post-audit descriptive replay of the preregistered distance-estimator metric using the actual serialized indexes; no settings or gates changed. Exact target values are squared L2 in each orthogonally transformed space.','rows':rows}
(ROOT/'runs'/'DISTANCE_ERROR_ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'rows':len(rows),'methods':{mth:{'mae':float(np.mean([r['mae_squared_distance'] for r in rows if r['method']==mth])),'signed':float(np.mean([r['signed_squared_distance_error'] for r in rows if r['method']==mth]))} for mth in ('unrotated','shared','mirror','independent')}}))
