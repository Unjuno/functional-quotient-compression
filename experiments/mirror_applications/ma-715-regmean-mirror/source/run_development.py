from __future__ import annotations
import csv,hashlib,json,io,statistics,sys,time
from pathlib import Path
import torch
from sklearn.datasets import load_digits
from merger import (aggregate_stats,regmean_from_stats,solve_mirror_code,fit_frobenius_code,decode,
                    orthogonal_basis,serialize,replay_payload,regression_metrics,weighted_merge_error,
                    features_for_rotation,fit_ridge_classifier,split_indices,source_statistics,
                    make_alphas,regmean_for_alpha,basis_library_payload)

BASE=Path(__file__).resolve().parents[1];SOURCE=BASE/'source';ART=SOURCE/'artifacts';ART.mkdir(parents=True,exist_ok=True)
SEEDS=(71501,71502);RANKS=(1,2,3,4);D=65;OUT=10;RIDGE=.001;MODEL_RIDGE=.1


def prepare_world(seed:int,manifest_path:Path)->dict:
    data=load_digits()
    train_i,dev_i,audit_i=split_indices(data.target,seed)
    manifest={'world_seed':seed,'train_indices':train_i,'development_indices':dev_i,'audit_indices_locked':audit_i,
              'counts':{'train':len(train_i),'development':len(dev_i),'audit':len(audit_i)},
              'audit_values_read':False,'split_rule':'stratified 60/20/20; seed, then seed+1'}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    images=torch.tensor(data.images,dtype=torch.float32);labels=torch.tensor(data.target,dtype=torch.long)
    tr_img=images[train_i];tr_y=labels[train_i];dv_img=images[dev_i];dv_y=labels[dev_i]
    tr_x=[features_for_rotation(tr_img,k) for k in range(4)]
    dv_x=[features_for_rotation(dv_img,k) for k in range(4)]
    # W0 is the clean source function; four source functions share train identities only.
    w0=fit_ridge_classifier(tr_x[0],tr_y,MODEL_RIDGE)
    source_models=[fit_ridge_classifier(tr_x[k],tr_y,MODEL_RIDGE) for k in range(4)]
    grams,crosses=source_statistics(source_models,tr_x)
    source_deltas=torch.stack(source_models)-w0.unsqueeze(0)
    source_basis={r:orthogonal_basis(source_deltas,r) for r in RANKS}
    train_alpha=make_alphas(seed+1000,8)
    train_regmeans=[]
    for a in train_alpha:
        wr,_,_=regmean_for_alpha(grams,crosses,a,RIDGE);train_regmeans.append(wr.to(torch.float32))
    output_deltas=torch.stack(train_regmeans)-w0.unsqueeze(0)
    output_basis={r:orthogonal_basis(output_deltas,r) for r in RANKS}
    dev_alpha=make_alphas(seed+2000,16)
    # Paid merge-time sufficient statistics; audit examples are not sliced here.
    stats_payload={'schema':'MA-715/stats-v1','world_seed':seed,'domains':[0,1,2,3],
                   'grams':grams,'crosses':crosses,'ridge_lambda':RIDGE,'feature_dim':D,'source_model_ridge':MODEL_RIDGE}
    stats_path=ART/f'seed{seed}_regmean_statistics.pt';stats_info=serialize(stats_payload,stats_path)
    source_payload={'schema':'MA-715/source-models-v1','world_seed':seed,'w0':w0,'source_models':torch.stack(source_models)}
    source_info=serialize(source_payload,ART/f'seed{seed}_source_models.pt')
    return {'seed':seed,'w0':w0,'source_models':source_models,'source_deltas':source_deltas,'source_basis':source_basis,
            'output_basis':output_basis,'grams':grams,'crosses':crosses,'train_alpha':train_alpha,'dev_alpha':dev_alpha,
            'train_regmeans':train_regmeans,'dev_domains':dv_x,'dev_labels':dv_y,'stats_payload':stats_payload,
            'stats_info':stats_info,'source_info':source_info,'manifest':manifest}


def estimate_proxy(method:str,rank:int|None)->int:
    d=D;c=OUT; nsrc=4
    full=nsrc*(d*d+c*d)+d**3+c*d*d
    if method=='full_regmean':return full
    if method=='task_arithmetic':return nsrc*c*d
    r=rank or 1
    if method.startswith('mirror_stats'):
        return nsrc*(d*d+c*d)+r*c*d*d+r*r*d+r**3
    return full+r*c*d


def build_method_outputs(world:dict,alpha:torch.Tensor,request_id:int)->tuple[dict[str,torch.Tensor],dict[str,dict]]:
    w0=world['w0'];grams=world['grams'];crosses=world['crosses'];source=world['source_models']
    t0=time.perf_counter();wfull,g,c=regmean_for_alpha(grams,crosses,alpha,RIDGE);full=wfull.to(torch.float32);full_wall=time.perf_counter()-t0
    outputs={'full_regmean':full,'task_arithmetic':sum(float(alpha[i])*source[i] for i in range(4))}
    details={}
    details['full_regmean']={'rank':None,'merge_error':0.0,'compute_proxy':estimate_proxy('full_regmean',None),'wall':full_wall}
    details['task_arithmetic']={'rank':None,'merge_error':weighted_merge_error(outputs['task_arithmetic'],full,g),'compute_proxy':estimate_proxy('task_arithmetic',None),'wall':0.0}
    for r in RANKS:
        bsrc=world['source_basis'][r];bout=world['output_basis'][r]
        t=time.perf_counter();m=solve_mirror_code(w0,bsrc,g,c,RIDGE);wm=decode(w0,bsrc,m);wm=wm.to(torch.float32);tm=time.perf_counter()-t
        key=f'mirror_stats_r{r}';outputs[key]=wm;details[key]={'rank':r,'code':m,'merge_error':weighted_merge_error(wm,full,g),'compute_proxy':estimate_proxy('mirror_stats',r),'wall':tm}
        t=time.perf_counter();ms=fit_frobenius_code(w0,bsrc,full);ws=decode(w0,bsrc,ms).to(torch.float32);ts=time.perf_counter()-t
        key=f'source_projection_r{r}';outputs[key]=ws;details[key]={'rank':r,'code':ms,'merge_error':weighted_merge_error(ws,full,g),'compute_proxy':estimate_proxy('source_projection',r),'wall':full_wall+ts}
        t=time.perf_counter();mo=fit_frobenius_code(w0,bout,full);wo=decode(w0,bout,mo).to(torch.float32);to=time.perf_counter()-t
        key=f'output_pca_r{r}';outputs[key]=wo;details[key]={'rank':r,'code':mo,'merge_error':weighted_merge_error(wo,full,g),'compute_proxy':estimate_proxy('output_pca',r),'wall':full_wall+to}
    metrics={}
    for method,weight in outputs.items():
        acc,nll=regression_metrics(weight,world['dev_domains'],world['dev_labels'],alpha)
        metrics[method]={'accuracy':acc,'nll':nll,**details[method]}
    return outputs,metrics


def make_payloads(world:dict,all_outputs:dict[str,list[torch.Tensor]],all_codes:dict[str,list[torch.Tensor]],
                  alpha:torch.Tensor)->tuple[dict,dict]:
    seed=world['seed'];w0=world['w0'];libs={};packages={}
    stats_info=world['stats_info'];stats_payload=world['stats_payload']
    recipe={'request_ids':list(range(len(alpha))),'alpha':alpha}
    recipe_stream=io.BytesIO();torch.save(recipe,recipe_stream);recipe_bytes=len(recipe_stream.getvalue())
    for method,models in all_outputs.items():
        if method=='full_regmean':payload={'schema':'MA-715/inference-v1','experiment':'MA-715','world_seed':seed,'method':method,'request_ids':list(range(len(models))),'models':torch.stack(models)}
        elif method=='task_arithmetic':payload={'schema':'MA-715/inference-v1','experiment':'MA-715','world_seed':seed,'method':method,'request_ids':list(range(len(models))),'shared':{'source_models':torch.stack(world['source_models'])},'alpha':alpha}
        else:
            if method.startswith('mirror_stats'):basis=world['source_basis'][int(method.rsplit('r',1)[1])]
            elif method.startswith('source_projection'):basis=world['source_basis'][int(method.rsplit('r',1)[1])]
            else:basis=world['output_basis'][int(method.rsplit('r',1)[1])]
            payload=basis_library_payload(method,seed,w0,basis,torch.stack(all_codes[method]),len(models))
        path=ART/f'seed{seed}_{method}_K{len(models)}.pt';info=serialize(payload,path)
        replay=replay_payload(path,torch.stack(models))
        info['roundtrip_max_weight_difference']=replay
        libs[method]=info
        if method=='task_arithmetic':package={'source_models':torch.stack(world['source_models']),'merge_alpha':alpha}
        else:package={'statistics':stats_payload,'merge_recipe':recipe,'inference_payload':payload}
        stream=io.BytesIO();torch.save(package,stream);package_bytes=len(stream.getvalue())
        packages[method]={'total_merge_package_bytes':package_bytes,'statistics_payload_bytes':stats_info['bytes'] if method!='task_arithmetic' else 0,'merge_recipe_bytes':recipe_bytes,'inference_payload_bytes':info['bytes'],'package_sha256':hashlib.sha256(stream.getvalue()).hexdigest()}
    return libs,packages


def main():
    torch.set_num_threads(1);start=time.perf_counter()
    worlds=[prepare_world(s,SOURCE/f'split_manifest_seed{s}.json') for s in SEEDS]
    perworld={};tables=[];world_records=[]
    for world in worlds:
        seed=world['seed']; alphas=world['dev_alpha']
        names=['full_regmean','task_arithmetic']+[f'{kind}_r{r}' for r in RANKS for kind in ('mirror_stats','source_projection','output_pca')]
        outputs_by_method={name:[] for name in names};codes_by_method={name:[] for name in names if name not in ('full_regmean','task_arithmetic')}
        raw=[]
        for req,a in enumerate(alphas):
            outputs,metrics=build_method_outputs(world,a,req)
            for name,w in outputs.items():outputs_by_method[name].append(w)
            # Recreate compact methods' codes from their actual weights in the declared basis for storage.
            for r in RANKS:
                codes_by_method[f'mirror_stats_r{r}'].append(solve_mirror_code(world['w0'],world['source_basis'][r],*aggregate_stats(world['grams'],world['crosses'],a),RIDGE))
                codes_by_method[f'source_projection_r{r}'].append(fit_frobenius_code(world['w0'],world['source_basis'][r],outputs[f'source_projection_r{r}']))
                codes_by_method[f'output_pca_r{r}'].append(fit_frobenius_code(world['w0'],world['output_basis'][r],outputs[f'output_pca_r{r}']))
            for name,m in metrics.items():
                record={'world_seed':seed,'request_id':req,'alpha':a.tolist(),'method':name,**{k:(v.tolist() if isinstance(v,torch.Tensor) else v) for k,v in m.items()}}
                raw.append(record);tables.append(record)
        libs,packages=make_payloads(world,outputs_by_method,codes_by_method,alphas)
        perworld[str(seed)]={'statistics_payload':world['stats_info'],'source_model_payload':world['source_info'],'library_payloads':libs,'merge_packages':packages,
                             'mean_metrics':{name:{metric:statistics.mean(r[metric] for r in raw if r['method']==name) for metric in ('accuracy','nll','merge_error','compute_proxy','wall')} for name in names},
                             'request_count':len(alphas)}
        world_records.extend(raw)
    # Choose the lowest-byte shared Mirror rank that meets full-RegMean quality/storage on both worlds.
    candidates=[]
    for r in RANKS:
        key=f'mirror_stats_r{r}';ok=True
        for seed in map(str,SEEDS):
            m=perworld[seed]['mean_metrics'][key];f=perworld[seed]['mean_metrics']['full_regmean']
            b=perworld[seed]['library_payloads'][key]['bytes'];fb=perworld[seed]['library_payloads']['full_regmean']['bytes']
            if m['accuracy']<f['accuracy']-.01 or m['nll']>f['nll']+.05 or b>.70*fb:ok=False
        if ok:candidates.append(r)
    selected=min(candidates,key=lambda r:statistics.mean(perworld[s]['library_payloads'][f'mirror_stats_r{r}']['bytes'] for s in map(str,SEEDS))) if candidates else None
    quality_gate=selected is not None
    mirror_specific={};
    if selected is not None:
        key=f'mirror_stats_r{selected}'
        for seed in map(str,SEEDS):
            target=perworld[seed]['mean_metrics'][key];bytes_m=perworld[seed]['library_payloads'][key]['bytes']
            controls=[]
            for method,metrics in perworld[seed]['mean_metrics'].items():
                if method in ('task_arithmetic',) or method.startswith('source_projection_') or method.startswith('output_pca_'):
                    b=perworld[seed]['library_payloads'][method]['bytes']
                    if .95*bytes_m<=b<=1.05*bytes_m:controls.append((method,metrics,b))
            best=min(controls,key=lambda x:x[1]['nll']) if controls else None
            if best:
                name,cm,cb=best;acc_gain=(target['accuracy']-cm['accuracy'])*100;nll_gain=cm['nll']-target['nll']
                compute_ratio=target['compute_proxy']/max(cm['compute_proxy'],1)
                pass_specific=(acc_gain>=1.0 or nll_gain>=.05 or (abs(acc_gain)<=.5 and compute_ratio<=.70))
            else:
                name=None;acc_gain=float('nan');nll_gain=float('nan');compute_ratio=float('nan');pass_specific=False
            mirror_specific[seed]={'best_byte_near_control':name,'accuracy_gain_percentage_points':acc_gain,'nll_improvement':nll_gain,'compute_proxy_ratio':compute_ratio,'pass':pass_specific}
    all_pass=quality_gate and all(x['pass'] for x in mirror_specific.values())
    summary={'experiment_id':'MA-715','status':'DEVELOPMENT_COMPLETE_PENDING_GATE','world_seeds':list(SEEDS),'dataset':'sklearn.load_digits','sklearn_version':__import__('sklearn').__version__,
             'split_manifests':[f'source/split_manifest_seed{s}.json' for s in SEEDS],'train_models_per_world':5,'train_merge_requests':8,'dev_merge_requests_per_world':16,'audit_accessed':False,
             'rank_sweep':list(RANKS),'selected_common_rank':selected,'quality_storage_gate_pass':quality_gate,'mirror_specific_gate_by_world':mirror_specific,'all_development_gates_pass':all_pass,
             'decision':'PASS_DEVELOPMENT' if all_pass else 'FAIL','world_results':perworld,'total_development_wall_seconds':time.perf_counter()-start}
    (SOURCE/'development_raw.json').write_text(json.dumps(world_records,indent=2)+'\n')
    (SOURCE/'development_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (BASE/'RESULTS_CORE.csv').open('w',newline='') as f:
        fields=['world_seed','request_id','alpha','method','accuracy','nll','merge_error','compute_proxy','wall','inference_payload_bytes','statistics_payload_bytes','total_merge_package_bytes']
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for rec in tables:
            info=perworld[str(rec['world_seed'])]
            library=info['library_payloads'][rec['method']]
            package=info['merge_packages'][rec['method']]
            w.writerow({'world_seed':rec['world_seed'],'request_id':rec['request_id'],'alpha':json.dumps(rec['alpha'],separators=(',',':')),'method':rec['method'],'accuracy':f"{rec['accuracy']:.8g}",'nll':f"{rec['nll']:.10g}",'merge_error':f"{rec['merge_error']:.10g}",'compute_proxy':rec['compute_proxy'],'wall':f"{rec['wall']:.9g}",'inference_payload_bytes':library['bytes'],'statistics_payload_bytes':package['statistics_payload_bytes'],'total_merge_package_bytes':package['total_merge_package_bytes']})
    if all_pass:
        audit=run_audit(worlds,selected)
        summary['audit_accessed']=True;summary['audit_results']=audit
        (SOURCE/'development_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def run_audit(worlds,rank):
    # This branch executes only after every development gate has passed.
    result={}
    for world in worlds:
        data=load_digits();seed=world['seed'];idx=world['manifest']['audit_indices_locked']
        imgs=torch.tensor(data.images[idx],dtype=torch.float32);labels=torch.tensor(data.target[idx],dtype=torch.long)
        domains=[features_for_rotation(imgs,k) for k in range(4)]
        alphas=make_alphas(seed+3000,32);metrics=[]
        for a in alphas:
            full,g,c=regmean_for_alpha(world['grams'],world['crosses'],a,RIDGE);full=full.float()
            mirror_code=solve_mirror_code(world['w0'],world['source_basis'][rank],g,c,RIDGE);wm=decode(world['w0'],world['source_basis'][rank],mirror_code).float()
            full_acc,full_nll=regression_metrics(full,domains,labels,a);mirror_acc,mirror_nll=regression_metrics(wm,domains,labels,a)
            metrics.append({'full_regmean_accuracy':full_acc,'full_regmean_nll':full_nll,'mirror_accuracy':mirror_acc,'mirror_nll':mirror_nll})
        result[str(seed)]={'new_merge_requests':len(metrics),'metrics':metrics,'mean':{k:statistics.mean(r[k] for r in metrics) for k in metrics[0]}}
    return result

if __name__=='__main__':
    s=main();print(json.dumps({'decision':s['decision'],'selected_common_rank':s['selected_common_rank'],'quality_storage_gate_pass':s['quality_storage_gate_pass'],'mirror_specific_gate_by_world':s['mirror_specific_gate_by_world'],'audit_accessed':s.get('audit_accessed',False)},indent=2))
