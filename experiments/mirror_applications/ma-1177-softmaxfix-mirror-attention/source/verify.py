"""Read-only 120-checkpoint replay + deterministic single-run replication, numerical report consistency."""
import csv,json,hashlib,sys,unittest
from pathlib import Path
import torch
from core import *
from run import ROOT,RESULTS,CHECKPOINTS,file_sha,load_record,model_outputs,train_hard,train_distill,choose_batch_order,fp64_cross_entropy,fp64_distill_kl

def verify():
 torch.set_num_threads(1)
 manifest=json.loads((RESULTS/'SEALED_TRAIN.json').read_text())
 assert manifest['model_count']==120 and not manifest['fresh_or_ood_generated']
 assert manifest['prereg_sha256']==file_sha(ROOT/'PROTOCOL.json')
 for p,h in manifest['source_sha256'].items():assert file_sha(ROOT/p)==h,(p,'source changed after sealing')
 records=manifest['records']; assert len(records)==120
 ckpts={r['file'] for r in records}; assert len(ckpts)==120
 assert ckpts=={p.name for p in CHECKPOINTS.glob('*.pt')}
 assert len(set(r['sha256'] for r in records))==120
 rows=list(csv.DictReader((RESULTS/'RESULTS_CORE.csv').open()))
 assert len(rows)==120 and len({r['checkpoint_file'] for r in rows})==120
 mapped={r['checkpoint_file']:r for r in rows}
 abs_err_max=0.; fresh_err_max=0.;reloaded=0
 for rec in records:
  m=rec['meta'];model=load_record(rec); row=mapped[rec['file']]
  assert row['loss']==m['loss'] and row['kind']==m['kind']
  if m['lane']=='hard':
   dat=dataset(m['world'],m['seed'],'dev',256)
   eval_dev=fp64_cross_entropy(model_outputs(model,dat),dat['target'])
   abs_err_max=max(abs_err_max,abs(eval_dev-m['dev_ce_fp64']))
   fresh=dataset(m['world'],m['seed'],'fresh',768)
   eval_fresh=fp64_cross_entropy(model_outputs(model,fresh),fresh['target'])
   fresh_err_max=max(fresh_err_max,abs(eval_fresh-float(row['fresh_ce_fp64'])))
  else:
   teacher_rec=next(r for r in records if r['meta']['lane']=='hard' and r['meta']['world']==m['world'] and r['meta']['seed']==m['seed'] and r['meta']['kind']==m['kind'] and r['meta']['beta']==12 and r['meta']['loss']=='fixed_ce')
   teacher=load_record(teacher_rec)
   fresh=dataset(m['world'],m['seed'],'fresh',768)
   kl=fp64_distill_kl(model_outputs(model,fresh),model_outputs(teacher,fresh))
   fresh_err_max=max(fresh_err_max,abs(kl-float(row['fresh_teacher_student_fp64_kl'])))
  reloaded+=1
 assert fresh_err_max<1e-12 and abs_err_max<1e-12
 # Re-run only one paired hard train for exact weight determinism.
 rec=next(r for r in records if r['meta']['lane']=='hard' and r['meta']['world']=='householder_aligned' and r['meta']['seed']==1801 and r['meta']['kind']=='mirror' and r['meta']['beta']==12.0 and r['meta']['loss']=='fixed_ce')
 tr=dataset('householder_aligned',1801,'train',384);dev=dataset('householder_aligned',1801,'dev',256)
 optmodel,meta=train_hard('householder_aligned',1801,'mirror',12.0,'fixed_ce',tr,dev,choose_batch_order(1801,550,'hard'))
 original=load_record(rec)
 assert all(torch.equal(v,original.state_dict()[k]) for k,v in optmodel.state_dict().items()),'hard replay differs'
 # Re-run one distillation from that frozen teacher.
 recd=next(r for r in records if r['meta']['lane']=='distill' and r['meta']['world']=='householder_aligned' and r['meta']['seed']==1801 and r['meta']['kind']=='mirror' and r['meta']['eps']==1e-8 and r['meta']['loss']=='standard_soft_ce')
 stu,_=train_distill('householder_aligned',1801,'mirror',1e-8,'standard_soft_ce',tr,choose_batch_order(1801,150,'distill'),original.state_dict())
 orig=load_record(recd)
 assert all(torch.equal(v,orig.state_dict()[k]) for k,v in stu.state_dict().items()),'distill replay differs'
 report=json.loads((RESULTS/'ANALYSIS.json').read_text());g=report['gates']
 assert report['record_counts']=={'hard':72,'distill':48,'posthoc_DEV_only':48}
 assert g['numeric'] and g['mirror_distill_gate'] and not g['mirror_hard_ce_gate'] and not g['mirror_specific_gate'] and not g['architecture_adoption']
 data={"id":"MA-1177","replay_checkpoints":reloaded,"unique_checkpoint_shas":120,
       "max_dev_ce_fp64_replay_error":abs_err_max,"max_fresh_metric_replay_error":fresh_err_max,
       "single_hard_full_retrain_bitwise_weights":True,"single_distill_full_retrain_bitwise_weights":True,
       "manifest_sha256":file_sha(RESULTS/'SEALED_TRAIN.json'),"csv_sha256":file_sha(RESULTS/'RESULTS_CORE.csv'),
       "torch":torch.__version__,"verification_status":"PASS"}
 (RESULTS/'VERIFICATION.json').write_text(json.dumps(data,indent=2)+'\n')
 print(json.dumps(data,indent=2))
if __name__=='__main__':verify()
