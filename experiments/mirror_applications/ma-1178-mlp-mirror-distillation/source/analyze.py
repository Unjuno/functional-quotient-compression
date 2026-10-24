"""Post-Fresh descriptive paired analyses using preregistered objective thresholds."""
import json,statistics,csv,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'results'/'FULL_RESULTS.json').read_text())
rows=D['rows'];W=['mirror_related','film_related','independent'];seeds=[101,211,307,401,503];widths=[8,16,24];arms=['mirror','film','shared','independent'];index={(x['world'],x['seed'],x['width'],x['arm']):x for x in rows}

out={'id':'MA-1178','row_count':len(rows),'quality_frontier':{},'pairwise':{},'learning_capacity':{},'teacher_function_diversity':D['world_metadata']}
for world in W:
  out['quality_frontier'][world]={}
  for width in widths:
    out['quality_frontier'][world][str(width)]={}
    for arm in arms:
      xs=[index[(world,s,width,arm)] for s in seeds]
      out['quality_frontier'][world][str(width)][arm]={
        'median_fresh_nmse':statistics.median(x['fresh_nmse'] for x in xs),
        'median_ood_nmse':statistics.median(x['ood_nmse'] for x in xs),
        'min_fresh_nmse':min(x['fresh_nmse'] for x in xs),
        'max_fresh_nmse':max(x['fresh_nmse'] for x in xs),
        'median_params':statistics.median(x['student_params'] for x in xs),
        'median_payload_bytes':statistics.median(x['packet_bytes'] for x in xs),
        'median_wall_train_s':statistics.median(x['fit_wall_s'] for x in xs),
        'median_b128_us':statistics.median(x['latency_128_us'] for x in xs),
        'per_seed_nmse':[x['fresh_nmse'] for x in xs],
      }
    pairs=[]
    for s in seeds:
      a=index[(world,s,width,'mirror')]
      film=index[(world,s,width,'film')]
      indep=index[(world,s,width,'independent')]
      pairs.append({'seed':s,'mirror':a['fresh_nmse'],'film':film['fresh_nmse'],'independent':indep['fresh_nmse'],'rel_gain_vs_film':(film['fresh_nmse']-a['fresh_nmse'])/max(1e-12,film['fresh_nmse']),'rel_gain_vs_independent':(indep['fresh_nmse']-a['fresh_nmse'])/max(1e-12,indep['fresh_nmse'])})
    out['pairwise'][f'{world}_h{width}']={'records':pairs,'mirror_beats_film_10pct':sum(p['rel_gain_vs_film']>=.10 for p in pairs),'mirror_beats_independent_10pct':sum(p['rel_gain_vs_independent']>=.10 for p in pairs),'mirror_beats_both_10pct':sum(p['rel_gain_vs_film']>=.10 and p['rel_gain_vs_independent']>=.10 for p in pairs)}
  out['learning_capacity'][world]={
    'mirror_16_fresh_below_0p01':sum(index[(world,s,16,'mirror')]['fresh_nmse']<.01 for s in seeds),
    'mirror_24_fresh_below_0p01':sum(index[(world,s,24,'mirror')]['fresh_nmse']<.01 for s in seeds),
    'mirror_16_vs_equal_film_10pct':out['pairwise'][f'{world}_h16']['mirror_beats_film_10pct'],
    'mirror_16_vs_independent_10pct':out['pairwise'][f'{world}_h16']['mirror_beats_independent_10pct']}
out['oracle_mirror_exact']={'per_seed_dev_nmse':[D['world_metadata'][str(s)]['mirror_oracle_dev_nmse'] for s in seeds], 'max':max(D['world_metadata'][str(s)]['mirror_oracle_dev_nmse'] for s in seeds)}
out['preregistered_decisions']={
 'mechanism_exact':out['oracle_mirror_exact']['max']<1e-10,
 'distill_mirror_related_h16_at_least_4of5_nmse_lt_0p01':out['learning_capacity']['mirror_related']['mirror_16_fresh_below_0p01']>=4,
 'mirror_specific_h16_4of5_above_10pct_vs_film_and_independent':out['pairwise']['mirror_related_h16']['mirror_beats_both_10pct']>=4,
 'cross_world_8of15_10pct_gain_vs_film_h16':sum(out['pairwise'][f'{w}_h16']['mirror_beats_film_10pct'] for w in W)>=8,
 'practical_pareto_not_claimed_without_full_compute':False}
(ROOT/'results'/'ANALYSIS.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'oracle':out['oracle_mirror_exact'],'decisions':out['preregistered_decisions'],'frontier':{world:{w:{arm:round(out['quality_frontier'][world][str(w)][arm]['median_fresh_nmse'],6) for arm in arms} for w in widths} for world in W}},indent=2,ensure_ascii=False))
