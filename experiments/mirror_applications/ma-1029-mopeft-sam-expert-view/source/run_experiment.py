"""CPU SAM ViT-B feature caching, development tuning, and fresh-world PEFT comparison."""
import argparse,hashlib,io,json,os,random,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from segment_anything import sam_model_registry,SamPredictor
from adapters import DOMAINS,FACTORIES

E=Path(__file__).resolve().parents[1];R=E/'runs';CKPT=Path('/tmp/ma1029-sam/sam_vit_b_01ec64.pth');SAMROOT=Path('/tmp/ma1029-sam');SAMCOMMIT=os.popen(f'git -C {SAMROOT} rev-parse HEAD').read().strip()
DOM2ID={x:i for i,x in enumerate(DOMAINS)};METHODS=('zero_shot','independent_adapter','hard_shared_adapter','mirror_givens','film_control','mopeft_like')
LR_CANDIDATES=(3e-4,1e-3);UPDATES_PER_DOMAIN=20;TRAIN_STEPS=UPDATES_PER_DOMAIN*len(DOMAINS)
torch.set_num_threads(8)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_manifest(phase):return json.loads((R/f'data_manifest_{phase}.json').read_text())

def load_model():
    model=sam_model_registry['vit_b'](checkpoint=str(CKPT)).eval()
    for p in model.parameters():p.requires_grad_(False)
    pred=SamPredictor(model)
    return model,pred

def encode_rows(model,predictor,rows):
    cache={};wall=[]
    for row in rows:
        img=np.asarray(Image.open(E/'runs'/row['image']).convert('RGB'))
        fp=R/'feature_cache'/f"{row['image_sha256']}.pt";fp.parent.mkdir(parents=True,exist_ok=True)
        if fp.exists():
            feat=torch.load(fp,map_location='cpu',weights_only=True);wall.append(0.)
        else:
            transformed=predictor.transform.apply_image(img)
            x=torch.as_tensor(transformed,device='cpu').permute(2,0,1).contiguous()[None].float()
            t=time.perf_counter()
            with torch.no_grad():feat=model.image_encoder(model.preprocess(x)).detach().cpu().contiguous()
            wall.append(time.perf_counter()-t);torch.save(feat,fp)
        box=predictor.transform.apply_boxes(np.asarray([row['box']],dtype=np.float32),img.shape[:2])
        mask=np.load(E/'runs'/row['mask'])['mask'].astype(np.uint8)
        cache[row['image']]={'features':feat,'box':torch.as_tensor(box,dtype=torch.float32),'mask':torch.from_numpy(mask).float()[None,None],'row':row}
    return cache,wall

def get_embeddings(model,boxes):
    with torch.no_grad():
        sparse,dense=model.prompt_encoder(points=None,boxes=boxes,masks=None)
    pe=model.prompt_encoder.get_dense_pe()
    return sparse,dense,pe

def predict_logits(model,adapter,task,ex):
    feat=ex['features'];x=feat if adapter is None else adapter.apply(feat,task)
    sparse,dense,pe=get_embeddings(model,ex['box'])
    low,_=model.mask_decoder(image_embeddings=x,image_pe=pe,sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,multimask_output=False)
    return low

def loss_for(model,adapter,task,ex):
    logits=predict_logits(model,adapter,task,ex)
    target=F.interpolate(ex['mask'],size=logits.shape[-2:],mode='nearest')
    bce=F.binary_cross_entropy_with_logits(logits,target)
    prob=logits.sigmoid();dice=1-(2*(prob*target).sum()+1)/(prob.sum()+target.sum()+1)
    return bce+dice

def iou(pred,truth):
    inter=np.logical_and(pred,truth).sum();union=np.logical_or(pred,truth).sum()
    return float(inter/union) if union else 1.0

def boundary_f(pred,truth,tol=2):
    p=torch.as_tensor(pred,dtype=torch.float32)[None,None];g=torch.as_tensor(truth,dtype=torch.float32)[None,None]
    pe=F.avg_pool2d(p,3,1,1)==1;ge=F.avg_pool2d(g,3,1,1)==1;pb=(p>0)&~pe;gb=(g>0)&~ge
    k=2*tol+1;pd=F.max_pool2d(pb.float(),k,1,tol)>0;gd=F.max_pool2d(gb.float(),k,1,tol)>0
    pr=float((pb&gd).sum()/(pb.sum()+1e-8));rc=float((gb&pd).sum()/(gb.sum()+1e-8));return 2*pr*rc/(pr+rc+1e-8)

def evaluate(model,adapter,cache,rows):
    by={d:{'iou':[],'boundary_f':[],'pred':[],'truth':[]} for d in DOMAINS};wall=[]
    with torch.no_grad():
        for row in rows:
            task=DOM2ID[row['domain']];ex=cache[row['image']];t=time.perf_counter();logits=predict_logits(model,adapter,task,ex);wall.append(time.perf_counter()-t)
            pred=(logits[0,0].sigmoid().cpu().numpy()>=.5);truth=ex['mask'][0,0].numpy().astype(bool)
            by[row['domain']]['iou'].append(iou(pred,truth));by[row['domain']]['boundary_f'].append(boundary_f(pred,truth));by[row['domain']]['pred'].append(pred.astype(np.uint8));by[row['domain']]['truth'].append(truth.astype(np.uint8))
    metrics={d:{'miou':float(np.mean(v['iou'])),'boundary_f':float(np.mean(v['boundary_f'])),'n':len(v['iou'])} for d,v in by.items()}
    metrics['macro_miou']=float(np.mean([metrics[d]['miou'] for d in DOMAINS]));metrics['macro_boundary_f']=float(np.mean([metrics[d]['boundary_f'] for d in DOMAINS]))
    return metrics,by,wall

def train(model,method,cache,train_rows,lr,seed):
    if method=='zero_shot':return None,0.
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    adapter=FACTORIES[method]();opt=torch.optim.Adam(adapter.parameters(),lr=lr,weight_decay=1e-4)
    grouped={d:[] for d in DOMAINS}
    for row in train_rows:grouped[row['domain']].append(row)
    start=time.perf_counter();step=0
    for task_name in DOMAINS:
        task=DOM2ID[task_name];examples=grouped[task_name]
        if not examples:raise ValueError(f'missing training examples for {task_name}')
        for k in range(UPDATES_PER_DOMAIN):
            row=examples[k%len(examples)];ex=cache[row['image']];opt.zero_grad(set_to_none=True)
            loss=loss_for(model,adapter,task,ex);loss.backward();opt.step();step+=1
    return adapter,time.perf_counter()-start

def serialize_adapter(method,adapter,lr,seed):
    state={} if adapter is None else {k:v.detach().cpu().contiguous() for k,v in adapter.state_dict().items()}
    meta={'schema':'MA1029-SAM-feature-PEFT-v1','method':method,'world_seed':seed,'learning_rate':lr,'domains':list(DOMAINS),'sam_model':'SAM ViT-B','sam_checkpoint_sha256':sha(CKPT),'sam_source_commit':SAMCOMMIT,'adapter_placement':'final SAM image embedding (256 channels, 64x64)','optimizer_state_included':False}
    bio=io.BytesIO();torch.save({'metadata':meta,'state_dict':state},bio);return bio.getvalue(),meta

def save_predictions(world,method,by):
    for domain,vals in by.items():
        np.savez_compressed(R/f'pred_{world}_{method}_{domain}.npz',pred=np.stack(vals['pred']),truth=np.stack(vals['truth']))

def train_eval_one(model,method,cache,train_rows,eval_rows,lr,world,seed):
    adapter,trainwall=train(model,method,cache,train_rows,lr,seed)
    metrics,by,eval_wall=evaluate(model,adapter,cache,eval_rows)
    payload,meta=serialize_adapter(method,adapter,lr,seed);p=R/f'payload_{world}_{method}.pt';p.write_bytes(payload)
    save_predictions(world,method,by)
    runtime_files=list((SAMROOT/'segment_anything').rglob('*.py'))+[E/'source/adapters.py',E/'source/run_experiment.py',E/'PROTOCOL.json']
    source_bytes=sum(p.stat().st_size for p in runtime_files)
    full=CKPT.stat().st_size+source_bytes+p.stat().st_size
    params=0 if adapter is None else sum(x.numel() for x in adapter.parameters())
    mac=0 if adapter is None else (2*256*8*64*64*(2 if method=='mopeft_like' else 1))
    return {'world':world,'seed':seed,'method':method,'learning_rate':lr,'train_updates':TRAIN_STEPS if adapter is not None else 0,'train_examples':TRAIN_STEPS if adapter is not None else 0,'train_wall_s':trainwall,'macro_miou':metrics['macro_miou'],'macro_boundary_f':metrics['macro_boundary_f'],'per_domain':{d:{k:v for k,v in row.items() if k in ('miou','boundary_f','n')} for d,row in metrics.items() if d in DOMAINS},'mean_decoder_wall_s_per_image':float(np.mean(eval_wall)) if eval_wall else 0.,'adapter_parameters':params,'adapter_mac_proxy_per_image':mac,'adaptation_payload_bytes':len(payload),'adaptation_payload_sha256':hashlib.sha256(payload).hexdigest(),'deployment_source_bytes':source_bytes,'full_deploy_bytes':full,'prediction_files':{d:hashlib.sha256((R/f'pred_{world}_{method}_{d}.npz').read_bytes()).hexdigest() for d in DOMAINS},'metadata':meta}

def phase_dev():
    manifest=load_manifest('dev');rows=manifest['rows'];model,predictor=load_model();cache,encwall=encode_rows(model,predictor,rows)
    train_rows=[r for r in rows if r['split']=='train'];dev_rows=[r for r in rows if r['split']=='dev'];tuning={}
    baseline,_,_=evaluate(model,None,cache,dev_rows);tuning['zero_shot']={'learning_rate':None,'development_macro_miou':baseline['macro_miou']}
    for method in FACTORIES:
        candidates=[]
        for lr in LR_CANDIDATES:
            model_state,wall=train(model,method,cache,train_rows,lr,17029)
            metrics,_,_=evaluate(model,model_state,cache,dev_rows)
            candidates.append({'learning_rate':lr,'development_macro_miou':metrics['macro_miou'],'train_wall_s':wall})
        best=max(candidates,key=lambda x:(x['development_macro_miou'],-x['learning_rate']))
        tuning[method]={'candidates':candidates,'selected_learning_rate':best['learning_rate'],'selected_development_macro_miou':best['development_macro_miou']}
        print(json.dumps({'phase':'dev','method':method,'selected':tuning[method]},sort_keys=True),flush=True)
    payload={'schema':'MA1029-dev-freeze-v1','protocol_sha256':sha(E/'PROTOCOL.json'),'dev_seed':102900,'image_encoder_seconds':sum(encwall),'image_encoder_seconds_per_image':float(np.mean(encwall)),'selected':tuning,'audit_seeds':[102901,102902,102903],'frozen':True}
    (R/'frozen_settings.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps({'frozen_settings':'runs/frozen_settings.json','encoder_images':len(rows),'encoder_wall_s':sum(encwall)},sort_keys=True))

def phase_audit():
    freeze=R/'frozen_settings.json'
    if not freeze.exists():raise RuntimeError('development freeze missing; audit blocked')
    settings=json.loads(freeze.read_text());manifest=load_manifest('audit');rows=manifest['rows'];model,predictor=load_model();cache,encwall=encode_rows(model,predictor,rows)
    out=R/'audit_rows.jsonl';out.unlink(missing_ok=True)
    encoder_by_image={r['image']:t for r,t in zip(rows,encwall)}
    for world,seed in [('audit_1',102901),('audit_2',102902),('audit_3',102903)]:
        wr=[r for r in rows if r['world']==world];train_rows=[r for r in wr if r['split']=='train'];audit_rows=[r for r in wr if r['split']=='audit']
        for method in ('zero_shot',)+tuple(FACTORIES):
            lr=None if method=='zero_shot' else settings['selected'][method]['selected_learning_rate']
            res=train_eval_one(model,method,cache,train_rows,audit_rows,lr,world,seed);res['image_encoder_wall_s']=sum(encoder_by_image[r['image']] for r in wr)
            with out.open('a') as f:f.write(json.dumps(res,sort_keys=True)+'\n')
            print(json.dumps({k:v for k,v in res.items() if k not in ('metadata','prediction_files')},sort_keys=True),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','audit'],required=True);a=ap.parse_args()
    if a.phase=='dev':phase_dev()
    else:phase_audit()
if __name__=='__main__':main()
