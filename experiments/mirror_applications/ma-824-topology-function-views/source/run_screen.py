"""Exact compositional program execution and payload screen for MA-824."""
import csv,hashlib,json,time
from pathlib import Path
import torch
from model import METHODS,TOPOLOGIES,FUNCS,function_bank,make_inputs,payload_for,save_payload,execute,execute_payload

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'source'/'artifacts';ART.mkdir(exist_ok=True)
DEV=(82401,82402);FRESH=(82406,82407,82408)


def run_one(seed,split,method):
    bank=function_bank(seed);x=make_inputs(seed,split);path=ART/f'amended_{split}_{seed}_{method}.pt'
    payload_info=save_payload(path,method,seed,bank);payload_info['path']=str(path.relative_to(ROOT))
    payload=torch.load(path,map_location='cpu',weights_only=False)
    errors={};throughput={};start=time.perf_counter()
    with torch.no_grad():
      for tid,topology in enumerate(TOPOLOGIES):
       for fid,name in enumerate(FUNCS):
        key=f'{topology}:{name}';target=execute(x,bank,tid,fid)
        pred=execute_payload(x,payload,topology,fid)
        errors[key]=float((pred-target).square().mean())
        inp=x[:512]
        for _ in range(2):execute_payload(inp,payload,topology,fid)
        t=time.perf_counter()
        for _ in range(10):execute_payload(inp,payload,topology,fid)
        throughput[key]=512*10/(time.perf_counter()-t)
    wall=time.perf_counter()-start
    # Approximate two-node affine MACs; nonlinearities and graph dispatch are reported separately by wall time.
    active_macs={'mirror_factors':2*(4+4*1),'native_interpreter':2*(4+4*1),
                 'hard_shared_indices':2*(4+4*1),'independent_programs':2*(4+4*1)}[method]
    return {'seed':seed,'split':split,'method':method,'payload':payload_info,'mse_by_program':errors,
      'max_program_mse':max(errors.values()),'mean_program_mse':sum(errors.values())/len(errors),
      'heldout_pair_mse':errors['parallel:sine_mlp'],'examples_per_program':len(x),
      'updates':0,'active_macs_per_example':active_macs,'wall_seconds':wall,
      'throughput_examples_per_second':sum(throughput.values())/len(throughput)}


def main():
    torch.set_num_threads(1);rows=[]
    for split,seeds in [('development',DEV),('fresh',FRESH)]:
      for seed in seeds:
       for method in METHODS:
        row=run_one(seed,split,method);rows.append(row)
        print(split,seed,method,row['max_program_mse'],row['payload']['bytes'],row['heldout_pair_mse'],flush=True)
    gate=[]
    for seed in FRESH:
      by={r['method']:r for r in rows if r['seed']==seed and r['split']=='fresh'}
      m,n,i=by['mirror_factors'],by['native_interpreter'],by['independent_programs']
      gate.append({'seed':seed,'all_programs_exact':m['max_program_mse']<=1e-10,
        'mirror_vs_independent_bytes':m['payload']['bytes']<=.8*i['payload']['bytes'],
        'mirror_vs_native_bytes':m['payload']['bytes']<=.9*n['payload']['bytes'],
        'native_exact':n['max_program_mse']<=1e-10,'independent_exact':i['max_program_mse']<=1e-10})
    summary={'results':rows,'fresh_gate':gate,'audit_opened':False,'audit':[],
      'protocol':{'dev_seeds':DEV,'fresh_seeds':FRESH,'learning':'none','programs':4,'heldout_pair':'parallel:sine_mlp'}}
    (ROOT/'source'/'screen_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fields=['condition','world_or_seed','method','serialized_bytes','adapter_only_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note']
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
      for r in rows:
       w.writerow({'condition':r['split'],'world_or_seed':r['seed'],'method':r['method'],'serialized_bytes':r['payload']['bytes'],'adapter_only_bytes':r['payload']['bytes'],'train_tokens_or_examples':4*r['examples_per_program'],'optimizer_updates':r['updates'],'active_compute_proxy':r['active_macs_per_example'],'wall_time_s':r['wall_seconds'],'primary_metric':'max/mean program MSE','primary_value':f"{r['max_program_mse']:.10g};{r['mean_program_mse']:.10g}",'secondary_metric':'heldout pair MSE;examples/s','secondary_value':f"{r['heldout_pair_mse']:.10g};{r['throughput_examples_per_second']:.6f}",'status_note':'fixed functions; no optimizer training'})
    print(json.dumps({'fresh_gate':gate,'audit_opened':False},indent=2))


if __name__=='__main__':main()
