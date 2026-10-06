"""Outcome-selected late-LR diagnostic. Never replaces SM002 primary results."""
from pathlib import Path
import json,time
import torch
import sm002 as s
from run_sm002 import ROOT,configure,dump,master_stream

def train(model,world,n_private,data,targets,steps,cutoff,new_lr,folder,save_every=500):
    folder.mkdir(parents=True,exist_ok=True)
    opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01,foreach=False)
    curve=[];t0=time.process_time()
    for step in range(1,steps+1):
        if new_lr is not None and step==cutoff+1:
            for g in opt.param_groups:g['lr']=new_lr
        opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(data[step-1]),targets[step-1])
        if not torch.isfinite(loss):raise FloatingPointError('loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%save_every==0 or step==steps:
            raw=s.encode(model);name=f'step{step:05d}.sm02';(folder/name).write_bytes(raw)
            curve.append({'step':step,'lr':opt.param_groups[0]['lr'],'payload_sha256':s.sha(raw),
                          'checkpoint':name,'quality':s.evaluate(model,world,n_private)})
    return {'curve':curve,'steps':steps,'process_cpu_including_audit_seconds':time.process_time()-t0}

def main():
    configure();protocol=json.loads((ROOT/'DIAGNOSTIC_PROTOCOL.json').read_text());results=[]
    for key in protocol['selected_cases']:
        original=ROOT/'main'/'runs'/key;r=json.loads((original/'result.json').read_text())
        world=s.World(r['world']);data=master_stream(world,16,128);target=world.targets(data.reshape(-1,3)).reshape(data.shape[:2])
        for branch,lr in [('constant_replay',None),('late_lr_0.001',.001)]:
            model=s.decode((original/'initial.sm02').read_bytes());folder=ROOT/'diagnostics'/key/branch
            result=train(model,world,16,data,target,4000,3500,lr,folder)
            equal_prefix=[]
            for c in result['curve']:
                if c['step']<=3500 or branch=='constant_replay':
                    equal=(folder/c['checkpoint']).read_bytes()==(original/'snapshots'/c['checkpoint']).read_bytes()
                    equal_prefix.append(equal);assert equal,(key,branch,c['step'])
            raw=s.encode(model);q=s.evaluate(s.decode(raw),world,16,wrong_views=True)
            result.update({'id':key,'branch':branch,'quality':q,'all_required_prefixes_byte_exact':all(equal_prefix),
                           'equal_snapshot_count':len(equal_prefix),'serialized_bytes':len(raw),'payload_sha256':s.sha(raw),
                           'main_result_unmodified':True,'evidence':'post-hoc selected-case intervention'})
            dump(folder/'result.json',result);results.append(result)
            print(key,branch,'private',q['private_accuracy'],'common',q['common_accuracy'],flush=True)
    dump(ROOT/'LATE_LR_DIAGNOSTIC.json',{'protocol':protocol,'results':results,
         'original_primary_results_unchanged':True,'causal_scope':'Only the paired effect of late learning-rate reduction in these3 outcome-selected cases; not proof of universal schedule advantage or original failure mechanistic decomposition.',
         'all_controls_reproduce_byte_exact':all(x['all_required_prefixes_byte_exact'] for x in results)})
if __name__=='__main__':main()
