"""CoOp/structured prompt-view screen on frozen CLIP and CIFAR-10."""
from __future__ import annotations
import hashlib,json,os,pathlib,time
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('HF_DATASETS_OFFLINE','1')
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image,ImageEnhance,ImageFilter
from transformers import AutoTokenizer,CLIPImageProcessor,CLIPModel
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask

ROOT=pathlib.Path(__file__).resolve().parents[1];RUNS=ROOT/'runs';RUNS.mkdir(exist_ok=True)
MODEL='openai/clip-vit-base-patch32';DATASET='uoft-cs/cifar10';SEED=102401;NCTX=4;UPDATES=30;LR=.002;SHOTS=4;AUDIT_N=8
CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'];BASE=list(range(6));NOVEL=list(range(6,10));DOMAINS=['original','grayscale','brightness_0.65','gaussian_blur_radius_1.0']
torch.set_num_threads(4)

def rotate_context(ctx,phi):
    c=torch.cos(phi);s=torch.sin(phi);out=ctx.clone()
    for i,j in ((0,1),(2,3)):
        out[i]=c*ctx[i]-s*ctx[j];out[j]=s*ctx[i]+c*ctx[j]
    return out

def context_features(model,tokenizer,names,ctx=None):
    ids=[];bos=tokenizer.bos_token_id;eos=tokenizer.eos_token_id;pad=tokenizer.pad_token_id
    xids=tokenizer('x',add_special_tokens=False).input_ids
    if len(xids)!=1: raise ValueError('placeholder must tokenize to one token')
    for name in names:
        suffix=tokenizer(' '+name+'.',add_special_tokens=False).input_ids
        row=[bos]+xids*NCTX+suffix+[eos];row=row[:77]+[pad]*max(0,77-len(row));ids.append(row)
    input_ids=torch.tensor(ids,dtype=torch.long)
    emb=model.text_model.embeddings.token_embedding(input_ids)
    if ctx is not None: emb[:,1:1+NCTX]=ctx[None].expand(len(names),-1,-1)
    hidden=model.text_model.embeddings(input_ids=input_ids,inputs_embeds=emb)
    causal=_create_4d_causal_attention_mask((len(names),77),hidden.dtype,hidden.device)
    enc=model.text_model.encoder(inputs_embeds=hidden,causal_attention_mask=causal,return_dict=True)
    last=model.text_model.final_layer_norm(enc.last_hidden_state)
    eos_pos=(input_ids==eos).int().argmax(-1); pooled=last[torch.arange(len(names)),eos_pos]
    return F.normalize(model.text_projection(pooled),dim=-1)

def zero_shot_features(model,tokenizer,names):
    batch=tokenizer([f'a photo of a {n}.' for n in names],padding='max_length',max_length=77,truncation=True,return_tensors='pt')
    return F.normalize(model.get_text_features(**batch),dim=-1)

def prompt_payload(method,ctx,phase=None):
    arrays=[]
    if method=='independent_domain_coop':arrays=[('domain_contexts',ctx.detach().cpu().numpy())]
    elif method=='shared_coop':arrays=[('shared_context',ctx.detach().cpu().numpy())]
    elif method=='mirror_domain_phase':arrays=[('shared_context',ctx.detach().cpu().numpy()),('domain_phase',phase.detach().cpu().numpy())]
    elif method=='zero_shot':arrays=[]
    header={'schema':'MA1024-CLIP-prompt-v1','model':MODEL,'n_context_tokens':NCTX,'classes':CLASSES,'domains':DOMAINS,'template':'x x x x {class}.','postprocessor':'normalized CLIP cosine logits','tensors':[{'name':n,'shape':list(a.shape),'dtype':'float32-le'} for n,a in arrays]}
    raw=json.dumps(header,separators=(',',':'),sort_keys=True).encode()+b'\n'
    return raw+b''.join(np.asarray(a,dtype='<f4').tobytes(order='C') for _,a in arrays)

def domain_image(image,d):
    im=image.convert('RGB')
    if d=='grayscale':im=Image.merge('RGB',(im.convert('L'),)*3)
    elif d=='brightness_0.65':im=ImageEnhance.Brightness(im).enhance(.65)
    elif d=='gaussian_blur_radius_1.0':im=im.filter(ImageFilter.GaussianBlur(1.0))
    return im

def choose_indices(ds,classes,n_each,seed):
    labels=np.asarray(ds['label']);rng=np.random.default_rng(seed);out={}
    for c in classes:
        ids=np.flatnonzero(labels==c);rng.shuffle(ids);out[c]=ids[:n_each].tolist()
    return out

def get_assets():
    path=pathlib.Path(snapshot_download(MODEL,local_files_only=True));model=CLIPModel.from_pretrained(path,local_files_only=True).eval();tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True);processor=CLIPImageProcessor.from_pretrained(path,local_files_only=True)
    req=['config.json','model.safetensors','pytorch_model.bin','vocab.json','merges.txt','tokenizer.json','tokenizer_config.json','special_tokens_map.json','preprocessor_config.json']
    files=[]
    for name in req:
        f=path/name
        if f.exists():files.append({'name':name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    total=sum(f['bytes'] for f in files)
    return path,model,tokenizer,processor,files,total

def image_features(model,processor,records):
    images=[];labels=[];domains=[]
    for d,items in enumerate(records):
        for image,label in items:
            images.append(domain_image(image,DOMAINS[d]));labels.append(label);domains.append(d)
    result=torch.empty((len(images),512));t0=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(images),32):
            pixel=processor(images=images[start:start+32],return_tensors='pt').pixel_values
            f=model.get_image_features(pixel_values=pixel);result[start:start+len(pixel)]=F.normalize(f,dim=-1)
    return result,torch.tensor(labels),torch.tensor(domains),time.perf_counter()-t0

def initial_context(model,tokenizer):
    ids=tokenizer('a photo of a cat.',return_tensors='pt').input_ids
    return model.text_model.embeddings.token_embedding(ids)[0,1:1+NCTX].detach().clone()

def contexts_for(kind,params,d):
    if kind=='shared_coop':return params['ctx']
    if kind=='independent_domain_coop':return params['ctx'][d]
    return rotate_context(params['ctx'],params['phase'][d])

def train_variant(kind,model,tokenizer,train_feats,train_labels,init):
    torch.manual_seed(SEED)
    if kind=='shared_coop':params={'ctx':torch.nn.Parameter(init.clone())}
    elif kind=='independent_domain_coop':params={'ctx':torch.nn.Parameter(init[None].repeat(len(DOMAINS),1,1).clone())}
    else:params={'ctx':torch.nn.Parameter(init.clone()),'phase':torch.nn.Parameter(torch.zeros(len(DOMAINS)))}
    opt=torch.optim.Adam(list(params.values()),lr=LR);t0=time.perf_counter();log=[]
    model.eval()
    for step in range(UPDATES):
        opt.zero_grad();loss=0.
        for d in range(len(DOMAINS)):
            txt=context_features(model,tokenizer,[CLASSES[i] for i in BASE],contexts_for(kind,params,d))
            logits=100.*train_feats[d]@txt.T
            loss=loss+F.cross_entropy(logits,train_labels[d])
        loss=loss/len(DOMAINS);loss.backward();opt.step();log.append(float(loss.detach()))
    return {k:v.detach() for k,v in params.items()},time.perf_counter()-t0,log

def eval_variant(kind,params,model,tokenizer,features,labels,domains):
    t0=time.perf_counter();stats={};by_domain={}
    for d,name in enumerate(DOMAINS):
        ctx=None if kind=='zero_shot' else contexts_for(kind,params,d)
        alltext=context_features(model,tokenizer,CLASSES,ctx) if ctx is not None else zero_shot_features(model,tokenizer,CLASSES)
        mask=(domains==d);y=labels[mask];f=features[mask];pred=(100.*f@alltext.T).argmax(-1)
        for group,ids in [('base',BASE),('novel',NOVEL)]:
            g=torch.isin(y,torch.tensor(ids));acc=float((pred[g]==y[g]).float().mean());by_domain[f'{name}_{group}']=acc;stats.setdefault(group,[]).append(acc)
    base=float(np.mean(stats['base']));novel=float(np.mean(stats['novel']));h=2*base*novel/(base+novel+1e-12)
    return {'base_accuracy':base,'novel_accuracy':novel,'harmonic_mean':h,'by_domain':by_domain,'prompt_inference_wall_s':time.perf_counter()-t0}

def run():
    t0=time.perf_counter();path,model,tokenizer,processor,files,backbone_bytes=get_assets();model.requires_grad_(False);model.eval()
    train=load_dataset(DATASET,split='train')
    train_ids=choose_indices(train,BASE,SHOTS,SEED)
    train_records=[[(train[int(i)]['img'],c) for c,ids in train_ids.items() for i in ids] for _ in DOMAINS]
    init_images,init_labels,init_domains,feature_train_wall=image_features(model,processor,train_records)
    train_feats=[init_images[(init_domains==d)] for d in range(len(DOMAINS))];train_lbls=[init_labels[(init_domains==d)] for d in range(len(DOMAINS))]
    init=initial_context(model,tokenizer);trained={};train_wall={};losses={}
    for kind in ('shared_coop','independent_domain_coop','mirror_domain_phase'):
        trained[kind],train_wall[kind],losses[kind]=train_variant(kind,model,tokenizer,train_feats,train_lbls,init)
    # The official CIFAR test split is accessed only after all prompt states are frozen.
    test=load_dataset(DATASET,split='test');test_ids=choose_indices(test,list(range(10)),AUDIT_N,SEED+1)
    audit_records=[[(test[int(i)]['img'],c) for c,ids in test_ids.items() for i in ids] for _ in DOMAINS]
    audit_features,audit_labels,audit_domains,feature_audit_wall=image_features(model,processor,audit_records)
    variants={'zero_shot':{},**trained,'native_scalar_givens':trained['mirror_domain_phase']};results={};prompt_bytes={};payloads={}
    for kind,params in variants.items():
        eval_kind='mirror_domain_phase' if kind=='native_scalar_givens' else kind
        results[kind]=eval_variant(eval_kind,params,model,tokenizer,audit_features,audit_labels,audit_domains)
        if eval_kind=='zero_shot':payload=prompt_payload('zero_shot',None)
        else:payload=prompt_payload(eval_kind,params.get('ctx'),params.get('phase'))
        payloads[kind]=payload;prompt_bytes[kind]=len(payload)
    results['native_scalar_givens']=dict(results['mirror_domain_phase'])
    payload_hashes={}
    for kind,payload in payloads.items():
        (RUNS/f'payload_{kind}.bin').write_bytes(payload);payload_hashes[kind]=hashlib.sha256(payload).hexdigest()
    full_bytes={k:backbone_bytes+v for k,v in prompt_bytes.items()}
    d={'schema':'MA1024-dev-v1','model':MODEL,'model_path':str(path),'model_files':files,'backbone_bytes':backbone_bytes,'backbone_parameters':sum(p.numel() for p in model.parameters()),'dataset':DATASET,'dataset_fingerprints':{'train':train._fingerprint,'test':test._fingerprint},'train_indices':train_ids,'audit_indices':test_ids,'audit_image_hashes':{str(c):[hashlib.sha256(test[int(i)]['img'].tobytes()).hexdigest() for i in ids] for c,ids in test_ids.items()},'fresh_opened':True,'fresh_opened_after_prompt_freeze':True,'fresh_seed_locked':True,'updates':UPDATES,'train_examples_per_domain':len(train_lbls[0]),'audit_examples_per_domain':sum(len(v) for v in test_ids.values()),'feature_extraction_wall_s':{'train':feature_train_wall,'audit':feature_audit_wall},'training_wall_s':train_wall,'training_loss_first_last':{k:[v[0],v[-1]] for k,v in losses.items()},'results':results,'prompt_payload_bytes':prompt_bytes,'prompt_payload_sha256':payload_hashes,'full_bundle_bytes':full_bytes,'mirror_native_payload_identical':payloads['mirror_domain_phase']==payloads['native_scalar_givens'],'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'total_wall_s':time.perf_counter()-t0,'boundaries':['CIFAR-10 only','four deterministic transforms','frozen CLIP ViT-B/32','small CPU study','not broad VLM generalization evidence']}
    (RUNS/'development.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
if __name__=='__main__':run()
