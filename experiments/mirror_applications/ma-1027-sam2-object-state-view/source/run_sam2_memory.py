"""Causal per-frame compression of official SAM2 object memory features."""
from __future__ import annotations
import hashlib,itertools,json,os,pathlib,time
os.environ.setdefault('SAM2_BUILD_CUDA','0')
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
E=pathlib.Path(__file__).resolve().parents[1];RUNS=E/'runs';FR=RUNS/'video_frames';GT=RUNS/'gt_masks'
REPO=pathlib.Path('/tmp/ma1027-sam2');CKPT=REPO/'sam2.1_hiera_tiny.pt';CONFIG=REPO/'sam2/configs/sam2.1/sam2.1_hiera_t.yaml'
MODEL='SAM2.1 Hiera-tiny';IMAGE_SIZE=256;OBJECT_IDS=[11,37,42,99];METHODS=['independent','hard_shared','mirror_givens','additive_rank1']
torch.set_num_threads(4)

def numpy_tensor(x,dtype=None):
    t=x.detach().cpu().contiguous()
    if t.dtype==torch.bfloat16:return t.view(torch.uint16).numpy().astype('<u2',copy=False)
    if dtype=='u1':return t.to(torch.uint8).numpy().astype('u1',copy=False)
    return t.float().numpy().astype('<f4',copy=False)

def array_bytes(name,array):return {'name':name,'array':np.ascontiguousarray(array)}

def givens(x,angles):
    # x [...,C,H,W], with independent rotations over pairs (0,1) and (2,3).
    y=x.clone()
    for k,(i,j) in enumerate(((0,1),(2,3))):
        c=torch.cos(angles[k]);s=torch.sin(angles[k]);a=x[...,i,:,:];b=x[...,j,:,:]
        y[...,i,:,:]=c*a-s*b;y[...,j,:,:]=s*a+c*b
    return y

def best_givens_angles(base,target):
    angles=[]
    for i,j in ((0,1),(2,3)):
        a=base[i].float();b=base[j].float();u=target[i].float();v=target[j].float()
        c=(a*u+b*v).sum();s=(a*v-b*u).sum();angles.append(torch.atan2(s,c))
    return torch.stack(angles)

def encode_memory(method,mems):
    # Input: [objects,1,C,H,W] bfloat16 memory features for one frame.
    t=mems.float().squeeze(1).cpu();n,c,h,w=t.shape;rep={'shape':[1,c,h,w],'dtype':'bfloat16'}
    base=t.mean(0).to(torch.bfloat16);basef=base.float()
    if method=='independent':
        rep['raw']=mems.squeeze(1).cpu().contiguous();decoded=rep['raw']
    elif method=='hard_shared':
        rep['base']=base;decoded=base[None].expand(n,-1,-1,-1).clone()
    elif method=='mirror_givens':
        angles=torch.stack([best_givens_angles(basef,z) for z in t]).float();rep['base']=base;rep['angles']=angles
        decoded=torch.stack([givens(basef,a) for a in angles]).to(torch.bfloat16)
    elif method=='additive_rank1':
        residual=t-basef;u,s,vh=torch.linalg.svd(residual.reshape(n,-1),full_matrices=False)
        coef=u[:,0]*s[0];basis=vh[0].reshape(c,h,w)
        pivot=torch.argmax(torch.abs(basis));sign=torch.sign(basis.flatten()[pivot]);sign=torch.where(sign==0,torch.ones_like(sign),sign)
        basis=(basis*sign).to(torch.bfloat16);coef=coef*sign
        rep.update({'base':base,'basis':basis,'coeff':coef.float()})
        decoded=(basef[None]+coef[:,None,None,None]*basis.float()[None]).to(torch.bfloat16)
    else:raise ValueError(method)
    return rep,decoded[:,None]

def decode_memory(method,rep):
    if method=='independent':return rep['raw'][:,None]
    base=rep['base'].float()
    if method=='hard_shared':return base[None,None].expand(len(OBJECT_IDS),-1,-1,-1,-1).to(torch.bfloat16).clone()
    if method=='mirror_givens':return torch.stack([givens(base,a) for a in rep['angles']]).to(torch.bfloat16)[:,None]
    if method=='additive_rank1':
        return (base[None]+rep['coeff'][:,None,None,None]*rep['basis'].float()[None]).to(torch.bfloat16)[:,None]
    raise ValueError(method)

def output_for_frame(state,obj_idx,frame):
    obj=state['output_dict_per_obj'][obj_idx]
    for key in ('cond_frame_outputs','non_cond_frame_outputs'):
        if frame in obj[key]:return obj[key][frame]
    raise KeyError((obj_idx,frame))

def bnd_f(pred,truth,tol=2):
    import torch.nn.functional as F
    p=torch.as_tensor(pred,dtype=torch.float32)[None,None];g=torch.as_tensor(truth,dtype=torch.float32)[None,None]
    pe=(F.avg_pool2d(p,3,1,1)==1).float();ge=(F.avg_pool2d(g,3,1,1)==1).float()
    pb=(p-pe)>0;gb=(g-ge)>0
    kernel=2*tol+1;pd=F.max_pool2d(pb.float(),kernel,1,tol)>0;gd=F.max_pool2d(gb.float(),kernel,1,tol)>0
    prec=float((pb&gd).sum()/(pb.sum()+1e-8));rec=float((gb&pd).sum()/(gb.sum()+1e-8))
    return 2*prec*rec/(prec+rec+1e-8)

def iou(a,b):
    inter=np.logical_and(a,b).sum();union=np.logical_or(a,b).sum();return float(inter/union) if union else 1.0

def best_assignment(pred,truth):
    n=len(pred);mat=np.array([[iou(pred[i],truth[j]) for j in range(n)] for i in range(n)])
    perm=max(itertools.permutations(range(n)),key=lambda q:sum(mat[i,q[i]] for i in range(n)))
    return list(perm),mat

def score_masks(pred_by_frame,truth_by_frame):
    js=[];fs=[];cross=[];assignments=[];switch=0;prev=None
    for f in range(len(pred_by_frame)):
        p=pred_by_frame[f];g=truth_by_frame[f];a,mat=best_assignment(p,g);assignments.append(a)
        if prev is not None:switch+=sum(a[i]!=prev[i] for i in range(len(a)))
        prev=a
        for i in range(len(p)):
            js.append(iou(p[i],g[i]));fs.append(bnd_f(p[i],g[i]))
            cross.extend(mat[i,j] for j in range(len(g)) if j!=i)
    # The initial-frame masks are exact user prompts and are not included in propagation quality.
    pj=[];pf=[];pc=[]
    for f in range(1,len(pred_by_frame)):
        for i,p in enumerate(pred_by_frame[f]):
            pj.append(iou(p,truth_by_frame[f][i]));pf.append(bnd_f(p,truth_by_frame[f][i]))
            pc.extend(iou(p,truth_by_frame[f][j]) for j in range(len(truth_by_frame[f])) if j!=i)
    return {'mean_j':float(np.mean(pj)),'mean_boundary_f':float(np.mean(pf)),'mean_jf':float((np.mean(pj)+np.mean(pf))/2),'id_switches':int(switch),'cross_object_iou':float(np.mean(pc)),'assignment_by_frame':assignments}

def arr_record(parts,name,a):parts.append(array_bytes(name,np.asarray(a)))

def serialize_parts(header,parts):
    arrays=[];manifest=[]
    for p in parts:
        a=p['array'];manifest.append({'name':p['name'],'shape':list(a.shape),'dtype':a.dtype.str,'bytes':int(a.nbytes)});arrays.append(a.tobytes(order='C'))
    header['tensors']=manifest;raw=json.dumps(header,sort_keys=True,separators=(',',':')).encode()+b'\n'+b''.join(arrays)
    return raw,manifest

def session_payload(method,rep_by_frame,state,all_frame_outputs,video_cache_payload,checkpoint_bytes,config_bytes,scene):
    parts=[];memory_bytes=0;private_bytes=0
    # User-provided first-frame masks and all private per-object outputs remain paid state.
    for obj_idx,oid in enumerate(OBJECT_IDS):
        mask=state['mask_inputs_per_obj'][obj_idx][0].detach().cpu().float().numpy().astype('<f4')
        arr_record(parts,f'private/prompt_mask/object_{oid}',mask);private_bytes+=mask.nbytes
        for frame in range(len(scene['rows'])):
            out=all_frame_outputs[(obj_idx,frame)]
            for key in ('pred_masks','obj_ptr','object_score_logits'):
                a=numpy_tensor(out[key]);arr_record(parts,f'private/object_{oid}/frame_{frame}/{key}',a);private_bytes+=a.nbytes
    for frame,rep in sorted(rep_by_frame.items()):
        for key,val in rep.items():
            if key=='shape' or key=='dtype':continue
            if isinstance(val,torch.Tensor):a=numpy_tensor(val)
            else:a=np.asarray(val,dtype='<f4')
            name=f'memory/frame_{frame}/{key}';arr_record(parts,name,a);memory_bytes+=a.nbytes
    h={'schema':'MA1027-session-v1','method':method,'model':MODEL,'checkpoint_sha256':hashlib.sha256(CKPT.read_bytes()).hexdigest(),'checkpoint_bytes':checkpoint_bytes,'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),'config_bytes':config_bytes,'scene_seed':102701,'frame_hashes':[r['jpeg_sha256'] for r in scene['rows']],'object_ids':OBJECT_IDS,'frames':len(scene['rows']),'image_size':IMAGE_SIZE,'view_mapping':{'method':method,'memory_feature_shape':next(iter(rep_by_frame.values()))['shape'],'feature_dtype':'bfloat16','per_frame_feature_bytes':int(np.prod(next(iter(rep_by_frame.values()))['shape'])*2)},'shared_video_cache_sha256':hashlib.sha256(video_cache_payload).hexdigest(),'shared_video_cache_bytes':len(video_cache_payload),'private_object_state_bytes':private_bytes,'encoded_memory_tensor_bytes':memory_bytes}
    raw,manifest=serialize_parts(h,parts);return raw,{'memory_tensor_bytes':memory_bytes,'private_object_state_bytes':private_bytes,'manifest':manifest,'header_bytes':len(raw)-sum(a['bytes'] for a in manifest)}

def run_method(predictor,method,scene,video_cache):
    state=predictor.init_state(str(FR),offload_video_to_cpu=False,offload_state_to_cpu=False)
    truth=[np.load(GT/f'{f:05}.npz')['masks'].astype(bool) for f in range(len(scene['rows']))]
    for obj_idx,oid in enumerate(OBJECT_IDS):predictor.add_new_mask(state,0,oid,torch.from_numpy(truth[0][obj_idx]))
    run_start=time.perf_counter();g=predictor.propagate_in_video(state);reps={};outputs={};predictions=[];encode_wall=0.;decode_wall=0.;prop_wall=0.
    while True:
        # Decode only for the next model step, then persist a factorized representation again at its yield.
        tdec=time.perf_counter()
        for fr,rep in reps.items():
            decoded=decode_memory(method,rep)
            for oi in range(len(OBJECT_IDS)):output_for_frame(state,oi,fr)['maskmem_features']=decoded[oi]
        decode_wall+=time.perf_counter()-tdec
        tmodel=time.perf_counter()
        try:f,ids,logits=next(g);prop_wall+=time.perf_counter()-tmodel
        except StopIteration:break
        pm=(logits.detach().cpu().numpy()[:,0]>0);predictions.append(pm)
        for oi in range(len(OBJECT_IDS)):outputs[(oi,int(f))]={k:v.detach().cpu().clone() for k,v in output_for_frame(state,oi,int(f)).items() if isinstance(v,torch.Tensor)}
        tenc=time.perf_counter();mems=torch.stack([output_for_frame(state,oi,int(f))['maskmem_features'] for oi in range(len(OBJECT_IDS))]);rep,decoded=encode_memory(method,mems);reps[int(f)]=rep
        if method!='independent':
            for oi in range(len(OBJECT_IDS)):output_for_frame(state,oi,int(f))['maskmem_features']=None
        encode_wall+=time.perf_counter()-tenc
    full_wall=time.perf_counter()-run_start
    score=score_masks(predictions,truth);state_raw,stat=session_payload(method,reps,state,outputs,video_cache,CKPT.stat().st_size,CONFIG.stat().st_size,scene)
    name=f'payload_{method}.bin';(RUNS/name).write_bytes(state_raw)
    np.savez_compressed(RUNS/f'predictions_{method}.npz',masks=np.stack(predictions).astype(np.uint8))
    return {'method':method,**score,'session_payload_bytes':len(state_raw),'object_memory_payload_bytes':stat['memory_tensor_bytes'],'private_object_state_bytes':stat['private_object_state_bytes'],'session_header_bytes':stat['header_bytes'],'memory_encode_wall_s':encode_wall,'memory_decode_wall_s':decode_wall,'encode_decode_wall_s':encode_wall+decode_wall,'propagation_wall_s':prop_wall,'full_run_wall_s':full_wall,'fps':len(predictions)/max(full_wall,1e-12),'payload_sha256':hashlib.sha256(state_raw).hexdigest(),'prediction_sha256':hashlib.sha256((RUNS/f'predictions_{method}.npz').read_bytes()).hexdigest(),'manifest':stat['manifest']}

def main():
    scene=json.loads((RUNS/'scene_manifest.json').read_text());
    for row in scene['rows']:
        f=FR/row['jpeg'];g=GT/f"{row['frame']:05}.npz";assert hashlib.sha256(f.read_bytes()).hexdigest()==row['jpeg_sha256'];assert hashlib.sha256(g.read_bytes()).hexdigest()==row['gt_sha256']
    checkpoint_bytes=CKPT.stat().st_size;config_bytes=CONFIG.stat().st_size;model_hash=hashlib.sha256(CKPT.read_bytes()).hexdigest()
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(CKPT),device='cpu',apply_postprocessing=False,hydra_overrides_extra=[f'++model.image_size={IMAGE_SIZE}'])
    # SAM2 caches normalized frames on CPU; serialize this one shared cache once and charge it to every full session.
    common=predictor.init_state(str(FR),offload_video_to_cpu=False,offload_state_to_cpu=False);cache=np.ascontiguousarray(common['images'].cpu().numpy().astype('<f4'));video_cache=json.dumps({'schema':'MA1027-frame-cache-v1','shape':list(cache.shape),'dtype':'float32-le'},sort_keys=True,separators=(',',':')).encode()+b'\n'+cache.tobytes(order='C');(RUNS/'shared_video_cache.bin').write_bytes(video_cache)
    # Discard the feasibility state's tensors before each fresh prompt session.
    del common
    results={}
    for method in METHODS:results[method]=run_method(predictor,method,scene,video_cache)
    common_session=checkpoint_bytes+config_bytes+len(video_cache)
    for r in results.values():r['full_session_payload_bytes']=common_session+r['session_payload_bytes']
    d={'schema':'MA1027-v1','prior_art':'PA315','official_repo':'https://github.com/facebookresearch/sam2','official_commit':'2b90b9f5ceec907a1c18123530e92e794ad901a4','model':MODEL,'checkpoint_path':str(CKPT),'checkpoint_bytes':checkpoint_bytes,'checkpoint_sha256':model_hash,'config_path':str(CONFIG),'config_bytes':config_bytes,'config_sha256':hashlib.sha256(CONFIG.read_bytes()).hexdigest(),'runtime':{'device':'cpu','torch':torch.__version__,'torchvision':__import__('torchvision').__version__,'image_size':IMAGE_SIZE,'threads':torch.get_num_threads(),'cuda_available':torch.cuda.is_available(),'cuda_extension_built':False},'scene':scene,'object_memory_private_state_included':['initial mask prompts','per-frame pred_masks','per-frame obj_ptr','per-frame object_score_logits'],'common_video_cache_bytes':len(video_cache),'common_video_cache_sha256':hashlib.sha256(video_cache).hexdigest(),'results':results,'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'scene_generator_sha256':hashlib.sha256((E/'source'/'generate_video.py').read_bytes()).hexdigest(),'boundaries':['one synthetic crossing scene','four objects','five frames','four propagated frames','SAM2.1 Hiera-tiny','CPU 256px','not natural-video benchmark evidence']}
    (RUNS/'development.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
if __name__=='__main__':main()
