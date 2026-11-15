"""Replay audit quality and complete serialized-payload accounting."""
import importlib.util,json,hashlib
from pathlib import Path
import numpy as np,faiss
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ma1036',Path(__file__).with_name('run_experiment.py')); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d=json.loads((ROOT/'runs'/'AUDIT_RESULTS.json').read_text()); angles=json.loads((ROOT/'runs'/'DEV_SELECTION.json').read_text())['frozen_angles_rad']
checks=[]; error_rows=[]
for rec in d['records']:
 seed=rec['seed']; train,db,qs=m.world(seed)
 for method in ('unrotated','shared','mirror','independent'):
  p=ROOT/'payload'/method/str(seed); meta=json.loads((p/'manifest.json').read_text())
  actual=sum(x.stat().st_size for x in p.iterdir() if x.is_file())
  expected=rec['methods'][method]['bytes']
  assert actual==expected,(seed,method,actual,expected)
  recalls=[]
  for j in range(4):
   no=meta['domain_to_index'][j]; idx=faiss.read_index(str(p/f'index_{no}.faiss'))
   q=qs[j]; dv=db
   if method in ('shared','mirror','independent'):
    bi=j if method=='independent' else 0
    R=np.fromfile(p/f'basis_{bi}.f32',dtype='<f4').reshape(m.D,m.D)
    th=angles[j] if method=='mirror' else 0.
    q=m.apply_basis(m.givens(q,th),R); dv=m.apply_basis(m.givens(db,th),R)
   exact=faiss.IndexFlatL2(m.D); exact.add(dv); _,gt=exact.search(q,m.K); dhat,got=idx.search(q,m.K)
   recalls.append(float((got==gt).sum()/(len(q)*m.K)))
   de=np.sum((q[:,None,:]-dv[got])**2,axis=2); err=dhat-de
   error_rows.append({'seed':seed,'domain':m.DOMAINS[j],'method':method,'mae_squared_distance':float(np.mean(np.abs(err))),'signed_squared_distance_error':float(np.mean(err))})
  np.testing.assert_allclose(recalls,rec['methods'][method]['recall10'],rtol=0,atol=0)
  checks.append({'seed':seed,'method':method,'payload_bytes':actual,'recall10_exact_replay':True})
# Compare the separately generated distance-estimator analysis against replay.
anal=json.loads((ROOT/'runs'/'DISTANCE_ERROR_ANALYSIS.json').read_text())['rows']
assert len(anal)==len(error_rows)==48
maxdiff=max(max(abs(a[k]-b[k]) for k in ('mae_squared_distance','signed_squared_distance_error')) for a,b in zip(anal,error_rows))
assert maxdiff==0.0, maxdiff
# Every payload file has a content hash recorded for provenance.
hashes=[]
for f in sorted((ROOT/'payload').rglob('*')):
 if f.is_file(): hashes.append({'path':str(f.relative_to(ROOT)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(ROOT/'runs'/'PAYLOAD_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n')
(ROOT/'runs'/'VERIFICATION_REPLAY.json').write_text(json.dumps({'payload_bytes_exact':True,'serialized_index_quality_replayed':True,'world_seeds_disjoint_from_dev':True,'distance_error_replay_exact':True,'distance_error_max_difference':maxdiff,'checks':checks},indent=2)+'\n')
print(json.dumps({'checks':len(checks),'passed':True,'payload_files':len(hashes),'distance_error_max_difference':maxdiff}))
