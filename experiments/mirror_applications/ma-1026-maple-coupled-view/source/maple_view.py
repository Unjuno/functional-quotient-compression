"""MaPLe-style coupled shallow/deep prompts with domain-specific paired views."""
from __future__ import annotations
import hashlib,json,os,pathlib,time
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('HF_DATASETS_OFFLINE','1')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image,ImageEnhance,ImageFilter
from transformers import AutoTokenizer,CLIPImageProcessor,CLIPModel
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
ROOT=pathlib.Path(__file__).resolve().parents[1];RUNS=ROOT/'runs';RUNS.mkdir(exist_ok=True)
MODEL='openai/clip-vit-base-patch32';DATASET='uoft-cs/cifar10';SEED=102601;AUDIT_SEED=102602;NCTX=2;DEPTH=2;UPDATES=20;BATCH=4;LR=.001
CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'];BASE=list(range(6));NOVEL=list(range(6,10));DOMAINS=['original','grayscale','brightness_0.65','gaussian_blur_radius_1.0'];torch.set_num_threads(4)

def choose(ds,classes,n,seed,excluded=None):
    labels=np.asarray(ds['label']);rng=np.random.default_rng(seed);out={};excluded=excluded or {}
    for c in classes:
        ids=np.flatnonzero(labels==c);rng.shuffle(ids);ban=set(map(int,excluded.get(str(c),[])));kept=[int(i) for i in ids if int(i) not in ban][:n]
        if len(kept)!=n:raise RuntimeError(f'insufficient disjoint examples for class {c}')
        out[str(c)]=kept
    return out

def domain_image(image,domain):
    im=image.convert('RGB')
    if domain=='grayscale':im=Image.merge('RGB',(im.convert('L'),)*3)
    elif domain=='brightness_0.65':im=ImageEnhance.Brightness(im).enhance(.65)
    elif domain=='gaussian_blur_radius_1.0':im=im.filter(ImageFilter.GaussianBlur(1.0))
    return im

def rotate_pair(x,angle):
    c=torch.cos(angle);s=torch.sin(angle);a=x[0];b=x[1]
    return torch.stack((c*a-s*b,s*a+c*b),dim=0)

class PromptState(nn.Module):
    def __init__(self,init_ctx,method,domains=len(DOMAINS)):
        super().__init__();self.method=method;self.domains=domains;torch.manual_seed(SEED)
        if method=='independent_domain_maple':
            self.ctx=nn.Parameter(init_ctx[None].repeat(domains,1,1).clone())
            self.proj=nn.ModuleList([nn.Linear(512,768) for _ in range(domains)])
            self.deep_text=nn.Parameter(torch.randn(domains,1,NCTX,512)*.02)
            self.deep_proj=nn.ModuleList([nn.Linear(512,768) for _ in range(domains)])
        else:
            self.ctx=nn.Parameter(init_ctx.clone());self.proj=nn.Linear(512,768)
            self.deep_text=nn.Parameter(torch.randn(1,NCTX,512)*.02);self.deep_proj=nn.Linear(512,768)
            if method in ('mirror_maple','additive_rank1_multimodal'):
                self.domain_codes=nn.Parameter(torch.zeros(domains,2))
            if method=='additive_rank1_multimodal':
                self.text_basis=nn.Parameter(torch.randn(DEPTH,NCTX,512)*.005)
                self.vision_basis=nn.Parameter(torch.randn(DEPTH,NCTX,768)*.005)

    def views(self,d):
        if self.method=='independent_domain_maple':
            tc=self.ctx[d];td=self.deep_text[d,0];vc=self.proj[d](tc);vd=self.deep_proj[d](td)
            return tc,td,vc,vd
        tc=self.ctx;td=self.deep_text[0];vc=self.proj(tc);vd=self.deep_proj(td)
        if self.method=='mirror_maple':
            mt=self.domain_codes[d,1];mv=self.domain_codes[d,0]
            tc=rotate_pair(tc,mt);td=rotate_pair(td,mt);vc=rotate_pair(vc,mv);vd=rotate_pair(vd,mv)
        elif self.method=='additive_rank1_multimodal':
            mv,mt=self.domain_codes[d]
            tc=tc+mt*self.text_basis[0];td=td+mt*self.text_basis[1]
            vc=self.proj(tc)+mv*self.vision_basis[0];vd=self.deep_proj(td)+mv*self.vision_basis[1]
        return tc,td,vc,vd

def get_assets():
    p=pathlib.Path(snapshot_download(MODEL,local_files_only=True));m=CLIPModel.from_pretrained(p,local_files_only=True).eval();t=AutoTokenizer.from_pretrained(p,local_files_only=True);pr=CLIPImageProcessor.from_pretrained(p,local_files_only=True);files=[]
    for name in ['config.json','model.safetensors','pytorch_model.bin','vocab.json','merges.txt','tokenizer.json','tokenizer_config.json','special_tokens_map.json','preprocessor_config.json']:
        f=p/name
        if f.exists():files.append({'name':name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    return p,m,t,pr,files,sum(q['bytes'] for q in files)

def initial_context(model,tok):
    ids=tok('a photo of a cat.',return_tensors='pt').input_ids
    return model.text_model.embeddings.token_embedding(ids)[0,1:1+NCTX].detach().clone()

def text_ids(tokenizer,names):
    xid=tokenizer('x',add_special_tokens=False).input_ids
    if len(xid)!=1:raise RuntimeError('x must be a single CLIP token')
    rows=[]
    for name in names:
        suf=tokenizer(' '+name+'.',add_special_tokens=False).input_ids
        row=([tokenizer.bos_token_id]+xid*NCTX+suf+[tokenizer.eos_token_id])[:77]
        row += [tokenizer.pad_token_id]*max(0,77-len(row));rows.append(row)
    return torch.tensor(rows,dtype=torch.long)

def encode_text_prompt(model,tok,names,tc,td):
    ids=text_ids(tok,names);tm=model.text_model;h=tm.embeddings(input_ids=ids);h=h.clone();h[:,1:1+NCTX]=tc[None].expand(len(ids),-1,-1)
    mask=_create_4d_causal_attention_mask((len(ids),77),h.dtype,h.device)
    for i,layer in enumerate(tm.encoder.layers):
        if i==1:h=h.clone();h[:,1:1+NCTX]=td[None].expand(len(ids),-1,-1)
        h=layer(h,attention_mask=None,causal_attention_mask=mask,output_attentions=False)[0]
    h=tm.final_layer_norm(h);eos=(ids==tok.eos_token_id).int().argmax(-1);pooled=h[torch.arange(len(ids)),eos]
    return F.normalize(model.text_projection(pooled),dim=-1)

def encode_image_prompt(model,pixels,vc,vd):
    vm=model.vision_model;h=vm.embeddings(pixel_values=pixels);h=torch.cat((h,vc[None].expand(len(pixels),-1,-1)),dim=1);h=vm.pre_layrnorm(h)
    for i,layer in enumerate(vm.encoder.layers):
        if i==1:h=h.clone();h[:,-NCTX:]=vd[None].expand(len(pixels),-1,-1)
        h=layer(h,attention_mask=None,causal_attention_mask=None,output_attentions=False)[0]
    pooled=vm.post_layernorm(h[:,0]);return F.normalize(model.visual_projection(pooled),dim=-1)

def zero_text(model,tok,names):
    x=tok([f'a photo of a {n}.' for n in names],padding='max_length',max_length=77,truncation=True,return_tensors='pt')
    return F.normalize(model.get_text_features(**x),dim=-1)

def data_records(ds,ids,heldout=True):
    out=[[] for _ in DOMAINS];hold=[]
    for c,arr in ids.items():
        ci=int(c)
        for d,name in enumerate(DOMAINS):
            pair=(ci,d)
            if heldout and d==ci%len(DOMAINS):hold.append({'class_id':ci,'domain_id':d});continue
            for ix in arr:out[d].append((ds[int(ix)]['img'],ci,int(ix)))
    return out,hold

def train_variant(method,model,tok,processor,records,init_ctx):
    net=PromptState(init_ctx,method);opt=torch.optim.Adam(net.parameters(),lr=LR);rng=np.random.default_rng(SEED);losses=[];t0=time.perf_counter();domain_counts={d:0 for d in range(len(DOMAINS))};model.eval()
    for step in range(UPDATES):
        d=step%len(DOMAINS);pool=records[d];ix=rng.choice(len(pool),BATCH,replace=False);batch=[pool[int(i)] for i in ix]
        ims=[domain_image(z[0],DOMAINS[d]) for z in batch];pixels=processor(images=ims,return_tensors='pt').pixel_values;ys=torch.tensor([BASE.index(z[1]) for z in batch])
        tc,td,vc,vd=net.views(d);opt.zero_grad(set_to_none=True)
        tf=encode_text_prompt(model,tok,[CLASSES[c] for c in BASE],tc,td);imf=encode_image_prompt(model,pixels,vc,vd)
        logits=model.logit_scale.exp().detach()*(imf@tf.T);loss=F.cross_entropy(logits,ys);loss.backward();opt.step();losses.append(float(loss.detach()));domain_counts[d]+=BATCH
    return net,time.perf_counter()-t0,{'first_last':[losses[0],losses[-1]],'examples_per_domain':domain_counts}

def summarize(preds,labels,domains,method_wall):
    correct={};counts={}
    for d in range(len(DOMAINS)):
        for c in range(len(CLASSES)):
            mask=(domains==d)&(labels==c);n=int(mask.sum());counts[(d,c)]=n;correct[(d,c)]=int((preds[mask]==c).sum())
    base=np.mean([correct[(d,c)]/counts[(d,c)] for d in range(len(DOMAINS)) for c in BASE]);novel=np.mean([correct[(d,c)]/counts[(d,c)] for d in range(len(DOMAINS)) for c in NOVEL]);h=2*base*novel/(base+novel+1e-12)
    hpairs=[]
    for c in BASE:
        d=c%len(DOMAINS);hpairs.append(correct[(d,c)]/counts[(d,c)])
    return {'base_accuracy':float(base),'novel_accuracy':float(novel),'harmonic_mean':float(h),'heldout_base_domain_pair_accuracy':float(np.mean(hpairs)),'by_domain':{DOMAINS[d]:{'base':float(np.mean([correct[(d,c)]/counts[(d,c)] for c in BASE])),'novel':float(np.mean([correct[(d,c)]/counts[(d,c)] for c in NOVEL]))} for d in range(len(DOMAINS))},'audit_wall_s':method_wall,'audit_wall_per_image_s':method_wall/max(1,sum(counts.values())),'cell_counts':{f'{DOMAINS[d]}:{CLASSES[c]}':counts[(d,c)] for d in range(len(DOMAINS)) for c in range(len(CLASSES))}} 

def evaluate(method,net,model,tok,processor,ds,audit_ids):
    rows=[];labels=[];domain_ids=[];t0=time.perf_counter();all_names=CLASSES
    with torch.inference_mode():
        for d,name in enumerate(DOMAINS):
            if method=='zero_shot':tf=zero_text(model,tok,all_names)
            else:tc,td,vc,vd=net.views(d);tf=encode_text_prompt(model,tok,all_names,tc,td)
            items=[(ds[int(ix)]['img'],int(c)) for c,arr in audit_ids.items() for ix in arr]
            for start in range(0,len(items),BATCH):
                batch=items[start:start+BATCH]; ims=[domain_image(im,name) for im,c in batch];pixels=processor(images=ims,return_tensors='pt').pixel_values
                if method=='zero_shot':imf=F.normalize(model.get_image_features(pixel_values=pixels),dim=-1)
                else:imf=encode_image_prompt(model,pixels,vc,vd)
                pr=(100.0*imf@tf.T).argmax(-1).tolist();rows.extend(pr);labels.extend([c for im,c in batch]);domain_ids.extend([d]*len(batch))
    return summarize(np.asarray(rows),np.asarray(labels),np.asarray(domain_ids),time.perf_counter()-t0)

def payload(net,method):
    arrays={k:v.detach().cpu().numpy().astype('<f4') for k,v in net.state_dict().items()};mapping={'shared_maple':'one coupled shallow/deep prompt set; deep prompt inserted after layer 0','independent_domain_maple':'one full coupled shallow/deep state and projections per domain','mirror_maple':'shared coupled states; separate per-domain text and vision Givens angles rotate the two prompt positions','additive_rank1_multimodal':'shared coupled states; per-domain text/vision scalars multiply rank-1 bases at shallow/deep prompt levels'}[method]
    h={'schema':'MA1026-selfdescribing-v1','method':method,'model':MODEL,'classes':CLASSES,'domains':DOMAINS,'n_context':NCTX,'prompt_depth':DEPTH,'token_template':'x x {class}.','image_transform_registry':DOMAINS,'score_scale':'frozen CLIP logit_scale.exp()','prompt_mapping':mapping,'tensor_manifest':[{'name':k,'shape':list(v.shape),'dtype':'float32-le'} for k,v in arrays.items()]}
    return json.dumps(h,sort_keys=True,separators=(',',':')).encode()+b'\n'+b''.join(v.tobytes(order='C') for v in arrays.values())

def main():
    start=time.perf_counter();p,model,tok,proc,files,backbone=get_assets();model.requires_grad_(False);model.eval();
    for x in model.parameters():x.requires_grad_(False)
    train=load_dataset(DATASET,split='train');test=load_dataset(DATASET,split='test')
    prev24=json.loads((ROOT.parent/'ma-1024-coop-prompt-mirror'/'runs'/'development.json').read_text());prev25=json.loads((ROOT.parent/'ma-1025-cocoop-instance-view'/'runs'/'development.json').read_text())
    ex_train={str(c):sorted(set(prev24['train_indices'][str(c)])|set(prev25['train_indices'][str(c)])) for c in BASE}
    train_ids=choose(train,BASE,4,SEED,ex_train)
    ex_test={str(c):sorted(set(prev24['audit_indices'][str(c)])|set(prev25['audit_indices'][str(c)])) for c in range(10)}
    audit_ids=choose(test,list(range(10)),4,AUDIT_SEED,ex_test)
    # Lock held-out Cartesian cells from train labels only; no test image pixels are accessed here.
    records,heldout_pairs=data_records(train,train_ids,True);init=initial_context(model,tok);trained={};train_wall={};losses={}
    for method in ('shared_maple','independent_domain_maple','mirror_maple','additive_rank1_multimodal'):
        trained[method],train_wall[method],losses[method]=train_variant(method,model,tok,proc,records,init)
    # Audit image pixels are accessed only after every trainable prompt state is frozen.
    audit_hash={c:[hashlib.sha256(test[int(ix)]['img'].tobytes()).hexdigest() for ix in ids] for c,ids in audit_ids.items()}
    results={'zero_shot':evaluate('zero_shot',None,model,tok,proc,test,audit_ids)}
    for method,net in trained.items():results[method]=evaluate(method,net,model,tok,proc,test,audit_ids)
    payloads={k:payload(v,k) for k,v in trained.items()};
    # Frozen zero-shot payload still charges class/domain and inference-template metadata.
    payloads['zero_shot']=(json.dumps({'schema':'MA1026-selfdescribing-v1','method':'zero_shot','model':MODEL,'classes':CLASSES,'domains':DOMAINS,'token_template':'a photo of a {class}.','score_scale':'frozen CLIP logit_scale.exp()','image_transform_registry':DOMAINS,'tensor_manifest':[]},sort_keys=True,separators=(',',':')).encode()+b'\n')
    for k,v in payloads.items():(RUNS/f'payload_{k}.bin').write_bytes(v)
    prompt_bytes={k:len(v) for k,v in payloads.items()};full={k:backbone+v for k,v in prompt_bytes.items()}
    d={'schema':'MA1026-v1','model':MODEL,'model_path':str(p),'model_files':files,'backbone_bytes':backbone,'backbone_parameters':sum(x.numel() for x in model.parameters()),'dataset':DATASET,'dataset_fingerprints':{'train':train._fingerprint,'test':test._fingerprint},'train_indices':train_ids,'audit_indices':audit_ids,'excluded_MA1024_MA1025_train_indices':ex_train,'excluded_MA1024_MA1025_test_indices':ex_test,'audit_image_sha256':audit_hash,'heldout_base_domain_pairs':heldout_pairs,'fresh_opened_after_all_prompt_states_frozen':True,'train_examples':sum(map(len,records)),'audit_images_per_class':4,'audit_examples_across_domains':len(CLASSES)*4*len(DOMAINS),'updates_per_method':UPDATES,'batch_size':BATCH,'train_wall_s':train_wall,'train_loss_first_last_and_domain_examples':losses,'results':results,'prompt_payload_bytes':prompt_bytes,'full_bundle_bytes':full,'payload_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in payloads.items()},'prompt_view_mac_proxy_per_image':{'mirror_givens_flops':6*(512+768)*DEPTH,'additive_rank1_macs':(512+768)*NCTX*DEPTH},'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'total_wall_s':time.perf_counter()-start,'boundaries':['CIFAR-10 only','frozen CLIP ViT-B/32','four deterministic image domains','2 prompt tokens and depth 2','20 updates per method, one seed','CPU screen, not official MaPLe reproduction']}
    (RUNS/'development.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
if __name__=='__main__':main()
