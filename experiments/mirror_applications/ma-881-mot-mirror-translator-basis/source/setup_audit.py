"""Post-fresh accounting audit for matrix-factor construction cost; no tuning."""
import json,time,statistics
from pathlib import Path
import engine
seeds=(88111,88112,88113)
shared=[];independent=[]
for seed in seeds:
 weights=engine.world(seed)[2]
 t=time.perf_counter();engine.shared_factor(weights,4);shared.append(time.perf_counter()-t)
 t=time.perf_counter();engine.independent_factor(weights,16);independent.append(time.perf_counter()-t)
# Thin-SVD arithmetic proxy. For m<=n, 4*m*m*n + 8*m*m*m.
def svd_proxy(m,n):
 if m>n:m,n=n,m
 return 4*m*m*n+8*m*m*m
out={
 'purpose':'Measure omitted compression setup cost after fresh quality/quality settings were already frozen; no hyperparameter or result selection changed.',
 'seeds':list(seeds),
 'shared_matrix_bank':{'svd_input_shape':[5,256],'calls_per_setup':1,'flop_proxy':svd_proxy(5,256),'wall_seconds':shared,'mean_wall_seconds':statistics.mean(shared)},
 'independent_rank16':{'svd_input_shapes_per_setup':[[16,16]]*5,'calls_per_setup':5,'flop_proxy':5*svd_proxy(16,16),'wall_seconds':independent,'mean_wall_seconds':statistics.mean(independent)},
 'native_mot_full':{'svd_calls':0,'flop_proxy':0},
 'mirror_givens_view':{'svd_calls':0,'flop_proxy':0,'note':'teacher angles were supplied; angle extraction/calibration is unmeasured'},
 'formula':'Thin-SVD proxy for m<=n: 4*m^2*n + 8*m^3; arithmetic estimate, not hardware counter.'
}
p=Path(__file__).resolve().parent/'SETUP_AUDIT.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
