"""Post-hoc, precision-controlled MA-1164B audit; no training or model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

WORLDS=(618460,618461,618462)
SEEDS=(41,42)
MODELS={
 256:('mirror','plain','narrow_3','rank_1','narrow_4','rank_2'),
 512:('mirror','plain','narrow_7','rank_4','narrow_8','rank_6'),
 768:('mirror','plain','narrow_10','rank_7','narrow_13','rank_10')}

def audit(rows):
    expected={'mirror':'weight_compile','plain':'weight_compile'}
    results=[]; paired=[]
    assert set(rows.world.unique())==set(WORLDS)
    assert set(rows.seed.unique())==set(SEEDS)
    assert rows.duplicated(['world','seed','m_target','family']).sum()==0
    for m, families in MODELS.items():
      part=rows[rows.m_target==m]
      assert set(part.family.unique())==set(families)
      for f in families:
        unit=part[part.family==f]
        assert len(unit)==6
        assert set(zip(unit.world,unit.seed))==set((w,s) for w in WORLDS for s in SEEDS)
        assert (unit.budget_mode==expected.get(f,'fixed900')).all()
        assert (unit.stored_bytes>0).all()
        assert (unit.accuracy.between(0,1)).all()
      assert part[part.family=='mirror'].stored_bytes.nunique()==1
      ref=part[part.family=='mirror'].set_index(['world','seed'])
      for family in families[1:]:
        comparator=part[part.family==family].set_index(['world','seed'])
        z=100*(ref.accuracy-comparator.accuracy)
        for (world,seed), val in z.items():
          paired.append({'m':m,'contrast':'mirror - '+family,'world':int(world),'seed':int(seed),'difference_pp':float(val),
           'mirror_accuracy':float(ref.loc[(world,seed),'accuracy']),'control_accuracy':float(comparator.loc[(world,seed),'accuracy'])})
        world_diffs=z.groupby(level=0).mean()
        results.append({'m':m,'contrast':'mirror - '+family,
          'mean_difference_pp':float(z.mean()),'world_mean_min_pp':float(world_diffs.min()),
          'world_mean_max_pp':float(world_diffs.max()),'paired_positive':int((z>0).sum()),
          'paired_negative':int((z<0).sum()),'paired_equal':int((z==0).sum()),
          'n_paired':len(z),'mirror_accuracy_mean':float(ref.accuracy.mean()),
          'control_accuracy_mean':float(comparator.accuracy.mean()),
          'mirror_bytes':int(ref.stored_bytes.iloc[0]),'control_bytes':int(comparator.stored_bytes.iloc[0]),
          'mirror_physical_scalars':m,'control_physical_scalars':int(comparator.learned_scalars.iloc[0]),
          'comparator_training_lane':'weight_compile' if family=='plain' else 'from_scratch_fixed900'})
    return pd.DataFrame(results), pd.DataFrame(paired)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='results/fp64_subset.csv')
    p.add_argument('--summary',default='results/paired_summary.csv')
    p.add_argument('--pairs',default='results/paired_detail.csv')
    p.add_argument('--verification',default='results/verification.json')
    args=p.parse_args()
    inp=Path(args.input); data=pd.read_csv(inp)
    summary,pairs=audit(data)
    summary.to_csv(args.summary,index=False,float_format='%.12g')
    pairs.to_csv(args.pairs,index=False,float_format='%.12g')
    verification={'status':'posthoc descriptive FP64-only paired analysis; no new learning',
      'FP64_model_coefficients_only':True,'source_subset_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),
      'data_rows':len(data),'summary_rows':len(summary),'paired_detail_rows':len(pairs),
      'worlds':list(WORLDS),'seeds':list(SEEDS),
      'rules':['exclude FP16 and FP32 rows','same world/seed pairing','do not count reused data partitions as independent domains',
               'do not treat teacher-weight compile as direct learning','no fresh hyperparameter adjustment']}
    Path(args.verification).write_text(json.dumps(verification,ensure_ascii=False,indent=2))
    print(summary[['m','contrast','mean_difference_pp','world_mean_min_pp','world_mean_max_pp']].to_string(index=False))

if __name__=='__main__':
    main()
