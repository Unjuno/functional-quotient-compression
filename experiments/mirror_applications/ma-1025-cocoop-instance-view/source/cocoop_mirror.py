"""Bounded image-conditioned CoCoOp/Mirror screen on frozen CLIP + CIFAR-10."""
from __future__ import annotations
import hashlib,json,os,pathlib,time
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('HF_DATASETS_OFFLINE','1')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer,CLIPImageProcessor,CLIPModel
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
ROOT=pathlib.Path(__file__).resolve().parents[1]; RUNS=ROOT/'runs'; RUNS.mkdir(exist_ok=True)
MODEL='openai/clip-vit-base-patch32'; DATASET='uoft-cs/cifar10'; SEED=102501; NCTX=4; HIDDEN=32; UPDATES=20; LR=.001; BATCH=4
CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']; BASE=list(range(6)); NOVEL=list(range(6,10)); torch.set_num_threads(4)

def choose(ds, classes, n, seed, excluded=None):
    labels=np.asarray(ds['label']); rng=np.random.default_rng(seed); out={}
    excluded=excluded or {}
    for c in classes:
        ids=np.flatnonzero(labels==c); rng.shuffle(ids); ban=set(map(int,excluded.get(str(c),[])))
        kept=[int(i) for i in ids if int(i) not in ban][:n]
        if len(kept)!=n: raise RuntimeError(f'not enough disjoint samples for class {c}')
        out[str(c)]=kept
    return out

def rotate(ctx, phi):
    # ctx [B,4,512], phi [B,2], independent Givens rotations over token pairs.
    out=ctx.clone()
    for k,(i,j) in enumerate(((0,1),(2,3))):
        c=torch.cos(phi[:,k])[:,None]; s=torch.sin(phi[:,k])[:,None]
        a=ctx[:,i]; b=ctx[:,j]; out[:,i]=c*a-s*b; out[:,j]=s*a+c*b
    return out

def text_batch(model, tok, names, contexts):
    # contexts [B,4,512], result [B,K,512]
    xids=tok('x',add_special_tokens=False).input_ids
    if len(xids)!=1: raise ValueError('x must be one token')
    ids=[]
    for name in names:
        suf=tok(' '+name+'.',add_special_tokens=False).input_ids
        row=([tok.bos_token_id]+xids*NCTX+suf+[tok.eos_token_id])[:77]
        row += [tok.pad_token_id]*max(0,77-len(row)); ids.append(row)
    K=len(names); B=contexts.shape[0]
    iid=torch.tensor(ids,dtype=torch.long).repeat(B,1)
    emb=model.text_model.embeddings.token_embedding(iid)
    emb[:,1:1+NCTX]=contexts[:,None].expand(B,K,NCTX,512).reshape(B*K,NCTX,512)
    hidden=model.text_model.embeddings(input_ids=iid,inputs_embeds=emb)
    mask=_create_4d_causal_attention_mask((B*K,77),hidden.dtype,hidden.device)
    enc=model.text_model.encoder(inputs_embeds=hidden,causal_attention_mask=mask,return_dict=True)
    last=model.text_model.final_layer_norm(enc.last_hidden_state)
    ep=(iid==tok.eos_token_id).int().argmax(-1); pooled=last[torch.arange(B*K),ep]
    return F.normalize(model.text_projection(pooled),dim=-1).reshape(B,K,-1)

def fixed_text(model,tok,names):
    batch=tok([f'a photo of a {n}.' for n in names],padding='max_length',max_length=77,truncation=True,return_tensors='pt')
    return F.normalize(model.get_text_features(**batch),dim=-1)

def get_assets():
    p=pathlib.Path(snapshot_download(MODEL,local_files_only=True)); m=CLIPModel.from_pretrained(p,local_files_only=True).eval(); t=AutoTokenizer.from_pretrained(p,local_files_only=True); proc=CLIPImageProcessor.from_pretrained(p,local_files_only=True)
    files=[]
    for n in ['config.json','model.safetensors','pytorch_model.bin','vocab.json','merges.txt','tokenizer.json','tokenizer_config.json','special_tokens_map.json','preprocessor_config.json']:
        f=p/n
        if f.exists(): files.append({'name':n,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    return p,m,t,proc,files,sum(x['bytes'] for x in files)

def feats(model,proc,ds,indices):
    ims=[]; ys=[]
    for c,arr in indices.items():
        for i in arr: ims.append(ds[int(i)]['img'].convert('RGB')); ys.append(int(c))
    x=torch.empty((len(ims),512)); t0=time.perf_counter()
    with torch.inference_mode():
        for s in range(0,len(ims),16):
            px=proc(images=ims[s:s+16],return_tensors='pt').pixel_values
            z=F.normalize(model.get_image_features(pixel_values=px),dim=-1); x[s:s+len(z)]=z
    return x,torch.tensor(ys),time.perf_counter()-t0

def init_ctx(model,tok):
    ids=tok('a photo of a cat.',return_tensors='pt').input_ids
    return model.text_model.embeddings.token_embedding(ids)[0,1:1+NCTX].detach().clone()

class Full(nn.Module):
    def __init__(self,ctx):
        super().__init__(); self.ctx=nn.Parameter(ctx.clone()); self.fc1=nn.Linear(512,HIDDEN); self.fc2=nn.Linear(HIDDEN,512)
    def forward(self,x): return self.ctx[None]+self.fc2(F.relu(self.fc1(x)))[:,None,:].expand(-1,NCTX,-1)
class Mir(nn.Module):
    def __init__(self,ctx):
        super().__init__(); self.ctx=nn.Parameter(ctx.clone()); self.fc1=nn.Linear(512,HIDDEN); self.fc2=nn.Linear(HIDDEN,2)
    def forward(self,x): return rotate(self.ctx[None].expand(len(x),-1,-1),self.fc2(F.relu(self.fc1(x))))
class Add(nn.Module):
    def __init__(self,ctx):
        super().__init__(); self.ctx=nn.Parameter(ctx.clone()); self.fc1=nn.Linear(512,HIDDEN); self.fc2=nn.Linear(HIDDEN,2); self.basis=nn.Parameter(torch.zeros(2,NCTX,512))
        nn.init.normal_(self.basis,std=.01)
    def forward(self,x):
        code=self.fc2(F.relu(self.fc1(x))); return self.ctx[None]+torch.einsum('bi,ijh->bjh',code,self.basis)
class Shared(nn.Module):
    def __init__(self,ctx): super().__init__(); self.ctx=nn.Parameter(ctx.clone())
    def forward(self,x): return self.ctx[None].expand(len(x),-1,-1)

def train(kind,model,tok,x,y,ctx):
    torch.manual_seed(SEED)
    net={'cocoop':Full,'mirror':Mir,'additive_rank2':Add,'shared_coop':Shared}[kind](ctx)
    opt=torch.optim.Adam(net.parameters(),lr=LR); losses=[]; t0=time.perf_counter(); rng=np.random.default_rng(SEED)
    for step in range(UPDATES):
        ix=rng.choice(len(y),size=BATCH,replace=False); xb=x[ix]; yb=y[ix]
        opt.zero_grad(); cf=net(xb); tf=text_batch(model,tok,[CLASSES[i] for i in BASE],cf)
        logits=100*torch.einsum('bd,bkd->bk',xb,tf); loss=F.cross_entropy(logits,torch.tensor([BASE.index(int(v)) for v in yb]))
        loss.backward(); opt.step(); losses.append(float(loss.detach()))
    return net, time.perf_counter()-t0, [losses[0],losses[-1]]

def eval_net(net,kind,model,tok,x,y,all_classes=True):
    names=CLASSES if all_classes else [CLASSES[i] for i in BASE]; t0=time.perf_counter(); correct={c:0 for c in range(10)}; counts={c:0 for c in range(10)}
    with torch.inference_mode():
        for s in range(0,len(y),4):
            xb=x[s:s+4]; yb=y[s:s+4]
            if kind=='zero': tf=fixed_text(model,tok,names)[None].expand(len(xb),-1,-1)
            else: tf=text_batch(model,tok,names,net(xb))
            pred=(100*torch.einsum('bd,bkd->bk',xb,tf)).argmax(-1)
            for j,c in enumerate(yb.tolist()):
                counts[c]+=1; correct[c]+=int(pred[j]==(names.index(CLASSES[c])))
    base=np.mean([correct[c]/counts[c] for c in BASE]); novel=np.mean([correct[c]/counts[c] for c in NOVEL]); h=2*base*novel/(base+novel+1e-12)
    return {'base_accuracy':float(base),'novel_accuracy':float(novel),'harmonic_mean':float(h),'per_image_wall_s':time.perf_counter()-t0,'counts':counts}

def state_bytes(net,kind):
    arrays={k:v.detach().cpu().numpy().astype('<f4') for k,v in net.state_dict().items()}
    mappings={
        'shared_coop':{'conditioning':'shared_context','activation':'none'},
        'cocoop':{'conditioning':'image_feature_to_512d_bias_repeated_over_context_tokens','mlp':'linear_512_32_relu_linear_32_512'},
        'mirror':{'conditioning':'image_feature_to_two_angles','mlp':'linear_512_32_relu_linear_32_2','view':'givens_rotation','token_pairs':[[0,1],[2,3]]},
        'additive_rank2':{'conditioning':'image_feature_to_two_coefficients','mlp':'linear_512_32_relu_linear_32_2','view':'rank2_additive_basis','basis_shape':[2,NCTX,512]}}
    header={'schema':'MA1025-payload-v1','method':kind,'model':MODEL,'classes':CLASSES,'token_template':'x x x x {class}.','placeholder_token':'x','n_ctx':NCTX,'hidden':HIDDEN,'score_scale':100.0,'image_feature_normalization':'L2','text_feature_normalization':'L2','conditioner_and_view':mappings[kind],'updates':UPDATES,'tensors':[{'name':k,'shape':list(a.shape),'dtype':'float32-le'} for k,a in arrays.items()]}
    return json.dumps(header,sort_keys=True,separators=(',',':')).encode()+b'\n'+b''.join(a.tobytes(order='C') for a in arrays.values())

def zero_shot_bytes():
    header={'schema':'MA1025-payload-v1','method':'zero_shot','model':MODEL,'classes':CLASSES,'token_template':'a photo of a {class}.','score_scale':100.0,'image_feature_normalization':'L2','text_feature_normalization':'L2','tensors':[]}
    return json.dumps(header,sort_keys=True,separators=(',',':')).encode()+b'\n'

def run():
    t0=time.perf_counter(); p,model,tok,proc,files,bbytes=get_assets(); model.requires_grad_(False); model.eval()
    ds=load_dataset(DATASET,split='train'); dt=load_dataset(DATASET,split='test')
    prev=json.loads((ROOT.parent/'ma-1024-coop-prompt-mirror'/'runs'/'development.json').read_text())
    train_ids=choose(ds,BASE,4,SEED,prev['train_indices']); trainx,trainy,trainwall=feats(model,proc,ds,train_ids)
    # audit subset excludes every previously opened test index; lock indices before feature extraction.
    audit_ids=choose(dt,list(range(10)),4,SEED+1,prev['audit_indices'])
    assert all(not(set(audit_ids[c]) & set(prev['audit_indices'][c])) for c in audit_ids)
    ctx=init_ctx(model,tok); nets={}; times={}; losses={}
    for kind in ('shared_coop','cocoop','mirror','additive_rank2'):
        nets[kind],times[kind],losses[kind]=train(kind,model,tok,trainx,trainy,ctx)
    # Zero-shot baseline; official test examples now opened after all learned state is frozen.
    auditx,audity,auditwall=feats(model,proc,dt,audit_ids)
    results={'zero_shot':eval_net(None,'zero',model,tok,auditx,audity)}
    for kind in nets: results[kind]=eval_net(nets[kind],kind,model,tok,auditx,audity)
    payloads={k:state_bytes(v,k) for k,v in nets.items()}; payloads['zero_shot']=zero_shot_bytes()
    for k,v in payloads.items(): (RUNS/f'payload_{k}.bin').write_bytes(v)
    prompt_bytes={k:len(v) for k,v in payloads.items()}; full={k:bbytes+n for k,n in prompt_bytes.items()}
    d={'schema':'MA1025-v1','model':MODEL,'model_path':str(p),'model_files':files,'backbone_bytes':bbytes,'backbone_parameters':sum(z.numel() for z in model.parameters()),'dataset':DATASET,'dataset_fingerprints':{'train':ds._fingerprint,'test':dt._fingerprint},'train_indices':train_ids,'audit_indices':audit_ids,'excluded_MA1024_train_indices':prev['train_indices'],'excluded_MA1024_audit_indices':prev['audit_indices'],'disjoint_from_MA1024':True,'fresh_opened':True,'fresh_opened_after_all_prompt_freeze':True,'seed':SEED,'updates':UPDATES,'batch_size':BATCH,'train_examples':len(trainy),'audit_examples':len(audity),'feature_wall_s':{'train':trainwall,'audit':auditwall},'training_wall_s':times,'loss_first_last':losses,'results':results,'prompt_payload_bytes':prompt_bytes,'full_bundle_bytes':full,'payload_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in payloads.items()},'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'total_wall_s':time.perf_counter()-t0,'boundaries':['CIFAR-10 only','one original image domain','frozen pretrained CLIP','small CPU study','disjoint image subset from MA-1024','not broad VLM generalization evidence']}
    (RUNS/'development.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
if __name__=='__main__':run()
