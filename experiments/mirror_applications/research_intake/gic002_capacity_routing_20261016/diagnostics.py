"""Descriptive diagnostics only; not a causal lock-in verdict."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from engine import ROOT,CFG

def main():
    groups=[];starved=0;seen=0;minimum=1.;pairs=[]
    for p in (ROOT/'results'/'fresh').glob('*/*.json'):
        obj=json.loads(p.read_text());rows=obj['records'];method=rows[0]['method']
        if method.endswith('oracle'):continue
        histories=obj['history'];uses=np.array([h['usage'] for h in histories]);lowest=np.minimum(uses,1-uses).min(0)
        starved+=int((lowest<.05).sum());seen+=len(rows);minimum=min(minimum,float(lowest.min()))
        first=np.abs(uses[0]-.5);last=np.array([r['test_mse'] for r in rows])
        corr=None if np.std(first)<1e-12 else float(spearmanr(first,last).statistic)
        groups.append({'world':rows[0]['world'],'family':rows[0]['family'],'ntrain':rows[0]['ntrain'],'method':method,'lowest_mean_usage':lowest.tolist(),'initial_imbalance_vs_test_spearman':corr})
    r={'logged_checkpoints':CFG['checkpoint_steps'],'nonoracle_runs':seen,'ever_below_5pct_mean_usage_at_logged_checkpoints':starved,'minimum_logged_mean_usage':minimum,'groups':groups,'causal_claim':False,'warning':'global mean usage does not rule out context-local or feature-level starvation; 8-seed correlations descriptive'}
    (ROOT/'DIAGNOSTICS.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k!='groups'})
if __name__=='__main__':main()
