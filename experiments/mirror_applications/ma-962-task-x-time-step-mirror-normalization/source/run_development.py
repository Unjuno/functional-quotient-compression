"""Fixed-budget MA-962 development run. Official test IDX files stay unopened until gates pass."""
from __future__ import annotations
import csv, hashlib, json, math, statistics, time
from pathlib import Path
import torch
from torch.nn import functional as F
from acquire_mnist import DATA, acquire_train, acquire_audit
from model import SharedSNN, IndependentTaskSNN, N_TASKS, TIMESTEPS, HIDDEN, PHASES, ENVELOPE, encode_rate_spikes, inference_macs_per_example

ROOT=Path(__file__).resolve().parents[1]; SOURCE=ROOT/'source'; ARTIFACTS=SOURCE/'artifacts'; ARTIFACTS.mkdir(exist_ok=True)
SEEDS=(96201,96202); CONDITIONS=('shared_no_task_gain','native_task_time_tebn','mirror_cyclic_phase','rank1_task_scalar_gate_on_shared_tebn_profile','independent_snn_per_task')
UPDATES=800; BATCH=32; EVAL_EVERY=100; EVAL_IMAGES=2048; LR=0.001; WD=0.0001

def load_train_dev():
    manifest=acquire_train()
    x=torch.load(DATA/'train_images.pt',map_location='cpu',weights_only=True).float()/255.0
    y=torch.load(DATA/'train_labels.pt',map_location='cpu',weights_only=True).long()
    return x[:50000],y[:50000],x[50000:],y[50000:],manifest

def selected_indices(n:int,count:int,seed:int):
    gen=torch.Generator().manual_seed(seed)
    return torch.randperm(n,generator=gen)[:count]

def batch_spikes(images,task_ids,seed):
    return encode_rate_spikes(images,task_ids,seed)

def eval_shared(model,images,labels,indices,seed):
    model.eval(); total_loss=0.; total=0; correct=0; spikes=0.; start=time.perf_counter()
    with torch.no_grad():
        for task in range(N_TASKS):
            for bi,offset in enumerate(range(0,len(indices),BATCH)):
                ix=indices[offset:offset+BATCH]; xb=images[ix]; yb=labels[ix]
                tasks=torch.full((len(ix),),task,dtype=torch.long)
                xs=batch_spikes(xb,tasks,seed+task*1000003+bi)
                logits,sc=model(xs,tasks); total_loss+=float(F.cross_entropy(logits,yb,reduction='sum'))
                correct+=int((logits.argmax(-1)==yb).sum()); total+=len(ix); spikes+=float(sc)*len(ix)
    elapsed=time.perf_counter()-start; model.train()
    return {'accuracy':correct/total,'loss':total_loss/total,'spikes_per_example':spikes/total,'evaluation_wall_seconds':elapsed,'inference_examples_per_second':total/max(elapsed,1e-12)}

def eval_independent(model,images,labels,indices,seed):
    total_loss=0.; total=0; correct=0; spikes=0.; start=time.perf_counter()
    for net in model.models: net.eval()
    with torch.no_grad():
        for task,net in enumerate(model.models):
            for bi,offset in enumerate(range(0,len(indices),BATCH)):
                ix=indices[offset:offset+BATCH]; xb=images[ix]; yb=labels[ix]
                tasks=torch.full((len(ix),),task,dtype=torch.long)
                xs=batch_spikes(xb,tasks,seed+task*1000003+bi)
                logits,sc=net(xs,tasks); total_loss+=float(F.cross_entropy(logits,yb,reduction='sum'))
                correct+=int((logits.argmax(-1)==yb).sum()); total+=len(ix); spikes+=float(sc)*len(ix)
    elapsed=time.perf_counter()-start
    for net in model.models: net.train()
    return {'accuracy':correct/total,'loss':total_loss/total,'spikes_per_example':spikes/total,'evaluation_wall_seconds':elapsed,'inference_examples_per_second':total/max(elapsed,1e-12)}

def payload(seed,condition,model,manifest):
    state={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    obj={'schema':'MA-962/inference-v1','world_seed':seed,'condition':condition,'state_dict':state,
         'config':{'input':784,'hidden':HIDDEN,'classes':10,'timesteps':TIMESTEPS,'leak':0.5,'threshold':1.0,
                   'tasks':4,'phase_offsets':PHASES.tolist(),'rate_envelope':ENVELOPE.tolist()},
         'dataset_train_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()}
    path=ARTIFACTS/f'seed{seed}_{condition}.pt'; torch.save(obj,path); raw=path.read_bytes()
    return {'path':str(path.relative_to(ROOT)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def make_shared(condition,seed):
    torch.manual_seed(seed)
    return SharedSNN(condition)

def train_shared(seed,condition,train_x,train_y,dev_x,dev_y,dev_idx,manifest):
    model=make_shared(condition,seed); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    plan_gen=torch.Generator().manual_seed(seed+12000)
    plan_i=torch.randint(0,len(train_x),(UPDATES,BATCH),generator=plan_gen)
    plan_t=torch.randint(0,N_TASKS,(UPDATES,BATCH),generator=plan_gen)
    best=None; best_acc=-1.; best_loss=float('inf'); best_step=0; start=time.perf_counter()
    for step in range(1,UPDATES+1):
        ix=plan_i[step-1]; tasks=plan_t[step-1]; xs=batch_spikes(train_x[ix],tasks,seed*100000+step)
        logits,_=model(xs,tasks); loss=F.cross_entropy(logits,train_y[ix])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%EVAL_EVERY==0:
            met=eval_shared(model,dev_x,dev_y,dev_idx,seed+400000)
            if met['accuracy']>best_acc or (met['accuracy']==best_acc and met['loss']<best_loss):
                best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_acc=met['accuracy']; best_loss=met['loss']; best_step=step
    train_s=time.perf_counter()-start; model.load_state_dict(best); final=eval_shared(model,dev_x,dev_y,dev_idx,seed+400000)
    return {'seed':seed,'condition':condition,'best_step':best_step,'development':final,'updates':UPDATES,'train_examples':UPDATES*BATCH,
            'train_wall_seconds':train_s,'active_macs_per_example':inference_macs_per_example(),'training_forward_backward_macs_proxy':inference_macs_per_example()*UPDATES*BATCH*3,
            'payload':payload(seed,condition,model,manifest)},model

def train_independent(seed,train_x,train_y,dev_x,dev_y,dev_idx,manifest):
    model=IndependentTaskSNN(seed+100); opts=[]; plans=[]
    for task,net in enumerate(model.models):
        opts.append(torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WD))
        gen=torch.Generator().manual_seed(seed+13000+task); plans.append(torch.randint(0,len(train_x),(UPDATES,BATCH),generator=gen))
    best_states=[None]*N_TASKS; best_acc=[-1.]*N_TASKS; best_loss=[float('inf')]*N_TASKS; best_step=[0]*N_TASKS
    start=time.perf_counter()
    for task,net in enumerate(model.models):
        for step in range(1,UPDATES+1):
            ix=plans[task][step-1]; tasks=torch.full((BATCH,),task,dtype=torch.long)
            xs=batch_spikes(train_x[ix],tasks,seed*100000+task*10000+step)
            logits,_=net(xs,tasks); loss=F.cross_entropy(logits,train_y[ix])
            opts[task].zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opts[task].step()
            if step%EVAL_EVERY==0:
                met=eval_shared_task(net,task,dev_x,dev_y,dev_idx,seed+400000)
                if met['accuracy']>best_acc[task] or (met['accuracy']==best_acc[task] and met['loss']<best_loss[task]):
                    best_states[task]={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}; best_acc[task]=met['accuracy']; best_loss[task]=met['loss']; best_step[task]=step
    train_s=time.perf_counter()-start
    for task,net in enumerate(model.models): net.load_state_dict(best_states[task])
    final=eval_independent(model,dev_x,dev_y,dev_idx,seed+400000)
    return {'seed':seed,'condition':'independent_snn_per_task','best_step':min(best_step),'best_step_by_task':best_step,'development':final,
            'updates':UPDATES*N_TASKS,'train_examples':UPDATES*BATCH*N_TASKS,'train_wall_seconds':train_s,
            'active_macs_per_example':inference_macs_per_example(),'training_forward_backward_macs_proxy':inference_macs_per_example()*UPDATES*BATCH*N_TASKS*3,
            'payload':payload(seed,'independent_snn_per_task',model,manifest)},model

def eval_shared_task(model,task,images,labels,indices,seed):
    model.eval(); loss=0.;total=0;correct=0; start=time.perf_counter()
    with torch.no_grad():
        for bi,off in enumerate(range(0,len(indices),BATCH)):
            ix=indices[off:off+BATCH]; xb=images[ix]; yb=labels[ix]; ts=torch.full((len(ix),),task,dtype=torch.long)
            xs=batch_spikes(xb,ts,seed+task*1000003+bi); logits,_=model(xs,ts)
            loss+=float(F.cross_entropy(logits,yb,reduction='sum'));correct+=int((logits.argmax(-1)==yb).sum());total+=len(ix)
    model.train();return {'accuracy':correct/total,'loss':loss/total,'evaluation_wall_seconds':time.perf_counter()-start}

def audit_gate_pass(summary):
    by=summary['per_seed']
    quality=all(by[str(s)]['mirror_cyclic_phase']['development']['accuracy'] >= by[str(s)]['native_task_time_tebn']['development']['accuracy']-0.015
                and by[str(s)]['mirror_cyclic_phase']['payload']['bytes'] <= .98*by[str(s)]['native_task_time_tebn']['payload']['bytes'] for s in SEEDS)
    specific=all(by[str(s)]['mirror_cyclic_phase']['development']['accuracy'] >= by[str(s)]['rank1_task_scalar_gate_on_shared_tebn_profile']['development']['accuracy']+0.01 for s in SEEDS)
    return quality,specific

def evaluate_audit(summary,models):
    test_x,test_y=acquire_audit(); results={}
    for seed,audit_seed in zip(SEEDS,(96203,96204)):
        idx=selected_indices(len(test_x),EVAL_IMAGES,audit_seed+700000);results[str(seed)]={}
        for cond,model in models[str(seed)].items():
            if cond=='independent_snn_per_task': met=eval_independent(model,test_x.float()/255.,test_y.long(),idx,audit_seed)
            else: met=eval_shared(model,test_x.float()/255.,test_y.long(),idx,audit_seed)
            results[str(seed)][cond]=met
    return results

def main():
    torch.set_num_threads(1)
    tx,ty,dx,dy,manifest=load_train_dev(); rows=[]; all_models={}; per_seed={}; wall=time.perf_counter()
    for seed in SEEDS:
        idx=selected_indices(len(dx),EVAL_IMAGES,seed+500000); per_seed[str(seed)]={}; all_models[str(seed)]={}
        for cond in CONDITIONS:
            if cond=='independent_snn_per_task': result,model=train_independent(seed,tx,ty,dx,dy,idx,manifest)
            else: result,model=train_shared(seed,cond,tx,ty,dx,dy,idx,manifest)
            per_seed[str(seed)][cond]=result; all_models[str(seed)][cond]=model
            d=result['development'];p=result['payload']
            rows.append({'world_seed':seed,'evaluation_split':'development','method':cond,'accuracy':d['accuracy'],'loss':d['loss'],
                'spikes_per_example':d['spikes_per_example'],'payload_bytes':p['bytes'],'payload_sha256':p['sha256'],
                'train_examples':result['train_examples'],'updates':result['updates'],'train_wall_seconds':result['train_wall_seconds'],
                'inference_examples_per_second':d['inference_examples_per_second'],'active_macs_per_example':result['active_macs_per_example'],
                'training_forward_backward_macs_proxy':result['training_forward_backward_macs_proxy'],'best_step':result['best_step']})
    summary={'experiment_id':'MA-962','dataset_manifest':manifest,'world_seeds':list(SEEDS),'conditions':list(CONDITIONS),'per_seed':per_seed,
             'development_only':True,'audit_accessed':False,'total_wall_seconds':time.perf_counter()-wall}
    quality,specific=audit_gate_pass(summary);summary['quality_storage_gate_pass']=quality;summary['mirror_specific_gate_pass']=specific
    summary['all_development_gates_pass']=quality and specific;summary['decision']='FAIL' if not (quality and specific) else 'PASS_DEVELOPMENT'
    (SOURCE/'development_raw.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    if quality and specific:
        summary['audit_accessed']=True; summary['audit_results']=evaluate_audit(summary,all_models)
        summary['dataset_manifest']=json.loads((DATA/'manifest.json').read_text())
        audit_ok=all(summary['audit_results'][str(s)]['mirror_cyclic_phase']['accuracy'] >= summary['audit_results'][str(s)]['native_task_time_tebn']['accuracy']-0.015
                      and summary['audit_results'][str(s)]['mirror_cyclic_phase']['accuracy'] >= summary['audit_results'][str(s)]['rank1_task_scalar_gate_on_shared_tebn_profile']['accuracy']+0.01 for s in SEEDS)
        summary['audit_gate_pass']=audit_ok; summary['decision']='PROMISING' if audit_ok else 'FAIL'
        audit_rows=[]
        for seed in SEEDS:
            for cond in CONDITIONS:
                met=summary['audit_results'][str(seed)][cond]; train_result=per_seed[str(seed)][cond]; pay=train_result['payload']
                audit_rows.append({'world_seed':seed,'evaluation_split':'audit','method':cond,'accuracy':met['accuracy'],'loss':met['loss'],
                    'spikes_per_example':met['spikes_per_example'],'payload_bytes':pay['bytes'],'payload_sha256':pay['sha256'],
                    'train_examples':train_result['train_examples'],'updates':train_result['updates'],'train_wall_seconds':train_result['train_wall_seconds'],
                    'inference_examples_per_second':met['inference_examples_per_second'],'active_macs_per_example':train_result['active_macs_per_example'],
                    'training_forward_backward_macs_proxy':train_result['training_forward_backward_macs_proxy'],'best_step':train_result['best_step']})
        with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writerows(audit_rows)
        (SOURCE/'audit_raw.json').write_text(json.dumps(audit_rows,indent=2)+'\n')
    summary['total_wall_seconds']=time.perf_counter()-wall
    (SOURCE/'development_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('decision','quality_storage_gate_pass','mirror_specific_gate_pass','audit_accessed','total_wall_seconds')},indent=2))
    return summary

if __name__=='__main__':main()
