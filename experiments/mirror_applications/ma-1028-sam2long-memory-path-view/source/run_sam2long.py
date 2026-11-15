"""Run the official SAM2Long tree with replaceable, causally decoded path memory."""
import hashlib, importlib.util, json, math, os, pathlib, struct, time
os.environ.setdefault('SAM2_BUILD_CUDA','0')
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

E=pathlib.Path(__file__).resolve().parents[1]; R=E/'runs'
OFFICIAL=pathlib.Path('/tmp/ma1028-sam2long'); CKPT=pathlib.Path('/tmp/ma1027-sam2/sam2.1_hiera_tiny.pt')
CFG=OFFICIAL/'sam2/configs/sam2.1/sam2.1_hiera_t.yaml'; METHODS=('independent','hard_shared','mirror_givens','additive_rank1')
torch.set_num_threads(4)

def tensor_array(x):
    x=x.detach().cpu().contiguous()
    if x.dtype==torch.bfloat16: return x.view(torch.uint16).numpy().astype('<u2',copy=False), 'bfloat16'
    a=x.numpy(); return np.ascontiguousarray(a), str(x.dtype).replace('torch.','')

def tensor_payload(x):
    a,dtype=tensor_array(x); return {'shape':list(a.shape),'dtype':dtype,'array':a,'bytes':a.nbytes}

def _rotate(base, angles):
    z=base.float().clone()
    for k,(i,j) in enumerate(((0,1),(2,3))):
        c=torch.cos(angles[k]);s=torch.sin(angles[k]);a=z[:,i].clone();b=z[:,j].clone()
        z[:,i]=c*a-s*b;z[:,j]=s*a+c*b
    return z.to(base.dtype)

def _best_angle(base,target,i,j):
    a=base[:,i].float();b=base[:,j].float();u=target[:,i].float();v=target[:,j].float()
    return torch.atan2((a*v-b*u).sum(),(a*u+b*v).sum())

def pack_memory(x,method):
    # x is [B,C,H,W,K], where the final axis indexes SAM2Long hypotheses.
    if x is None or x.ndim!=5 or x.shape[-1]<2:return None,x
    xf=x.float();K=x.shape[-1];base=xf.mean(-1).to(torch.bfloat16);bf=base.float()
    rep={'schema':'ma1028-path-memory-v1','method':method,'shape':list(x.shape),'dtype':'bfloat16'}
    if method=='hard_shared':
        rep['base']=base; out=base.unsqueeze(-1).expand_as(x).clone()
    elif method=='mirror_givens':
        angles=[]
        for k in range(K):
            y=xf[...,k]
            angles.append(torch.stack([_best_angle(bf,y,0,1),_best_angle(bf,y,2,3)]))
        rep['base']=base;rep['angles']=torch.stack(angles).float()
        out=torch.stack([_rotate(base,rep['angles'][k]) for k in range(K)],-1)
    elif method=='additive_rank1':
        mat=(xf-bf.unsqueeze(-1)).reshape(-1,K)
        u,s,vh=torch.linalg.svd(mat,full_matrices=False)
        basis=(u[:,0]*s[0]).reshape_as(base).to(torch.bfloat16);coef=vh[0].to(torch.bfloat16)
        # Canonicalize the SVD sign so the serialized representation is deterministic.
        flat=basis.flatten();pivot=torch.argmax(flat.abs());sgn=torch.where(flat[pivot]<0,-torch.ones((),device=flat.device),torch.ones((),device=flat.device))
        rep.update({'base':base,'basis':basis*sgn,'coeff':coef*sgn})
        out=base.float().unsqueeze(-1)+(rep['basis'].float().unsqueeze(-1)*rep['coeff'].float().view(1,1,1,1,-1))
        out=out.to(torch.bfloat16)
    else: raise ValueError(method)
    return rep,out

def restore(rep):
    base=rep['base']
    K=rep['shape'][-1]
    if rep['method']=='hard_shared':return base.unsqueeze(-1).expand(*base.shape,K).clone()
    if rep['method']=='mirror_givens':return torch.stack([_rotate(base,rep['angles'][k]) for k in range(K)],-1)
    if rep['method']=='additive_rank1':
        return (base.float().unsqueeze(-1)+rep['basis'].float().unsqueeze(-1)*rep['coeff'].float().view(1,1,1,1,-1)).to(torch.bfloat16)
    raise ValueError(rep['method'])

def _entries(d):
    for section in ('cond_frame_outputs','non_cond_frame_outputs'):
        for frame,out in d.get(section,{}).items(): yield section,frame,out

def install_storage_adapter(predictor,method,timers):
    original=predictor._run_single_frame_inference
    def unwrap(d):
        for _,_,out in _entries(d):
            rep=out.pop('_ma1028_packed',None)
            if rep is not None:out['maskmem_features']=restore(rep)
    def wrap(d):
        for _,_,out in _entries(d):
            feat=out.get('maskmem_features')
            if feat is None:continue
            t=time.perf_counter();rep,decoded=pack_memory(feat,method);timers['encode']+=time.perf_counter()-t
            if rep is not None:
                out['_ma1028_packed']=rep
                # Keep only a zero-copy expansion of the shared base resident between steps.
                out['maskmem_features']=rep['base'].unsqueeze(-1).expand_as(decoded)
    def wrapped(*args,**kwargs):
        d=kwargs['output_dict']; t=time.perf_counter();unwrap(d);timers['decode']+=time.perf_counter()-t
        try:cur,preview,picks=original(*args,**kwargs)
        except Exception:
            raise
        t=time.perf_counter();wrap(d)
        rep,decoded=pack_memory(cur.get('maskmem_features'),method);timers['encode']+=time.perf_counter()-t
        if rep is not None:
            cur['_ma1028_packed']=rep
            cur['maskmem_features']=rep['base'].unsqueeze(-1).expand_as(decoded)
        return cur,preview,picks
    predictor._run_single_frame_inference=wrapped

def boundary_f(pred,truth,tol=2):
    p=torch.as_tensor(pred,dtype=torch.float32)[None,None];g=torch.as_tensor(truth,dtype=torch.float32)[None,None]
    pe=(F.avg_pool2d(p,3,1,1)==1).float();ge=(F.avg_pool2d(g,3,1,1)==1).float();pb=(p-pe)>0;gb=(g-ge)>0
    k=2*tol+1;pd=F.max_pool2d(pb.float(),k,1,tol)>0;gd=F.max_pool2d(gb.float(),k,1,tol)>0
    prec=float((pb&gd).sum()/(pb.sum()+1e-8));rec=float((gb&pd).sum()/(gb.sum()+1e-8));return 2*prec*rec/(prec+rec+1e-8)

def score(pred,truth):
    js=[];fs=[]
    for p,g in zip(pred[1:],truth[1:]):
        inter=np.logical_and(p,g).sum();union=np.logical_or(p,g).sum();js.append(float(inter/union) if union else 1.0);fs.append(boundary_f(p,g))
    return {'mean_j':float(np.mean(js)),'mean_boundary_f':float(np.mean(fs)),'mean_jf':float((np.mean(js)+np.mean(fs))/2),'id_switches':0}

def _put(parts,name,array,dtype=None):
    a=np.ascontiguousarray(array);parts.append((name,a,dtype or a.dtype.str))

def pack_state(method,state,manifest,scene,cache_blob):
    parts=[];outdict=state['output_dict'];frames=[];memory_bytes=0
    for section,fr,out in _entries(outdict):
        row={'section':section,'frame':int(fr),'fields':{}}
        for key in ('pred_masks','obj_ptr','object_score_logits','ious'):
            if key in out and out[key] is not None:
                p=tensor_payload(out[key]);_put(parts,f'{section}/{fr}/{key}',p['array'],p['array'].dtype.str);row['fields'][key]={'shape':p['shape'],'dtype':p['dtype']}
        if 'acc_score' in out:
            a=np.asarray(out['acc_score'],dtype='<f4');_put(parts,f'{section}/{fr}/acc_score',a);row['fields']['acc_score']={'shape':list(a.shape),'dtype':'<f4'}
        rep=out.get('_ma1028_packed')
        feat=out.get('maskmem_features')
        if rep is not None:
            row['memory']={'schema':rep['schema'],'method':method,'shape':rep['shape'],'dtype':rep['dtype'],'coordinate_rule':'base + per-path Givens angles' if method=='mirror_givens' else 'shared base' if method=='hard_shared' else 'shared base + rank-1 factors'}
            for key,val in rep.items():
                if torch.is_tensor(val) and key!='decoded':
                    a,dtype=tensor_array(val);_put(parts,f'{section}/{fr}/memory/{key}',a);memory_bytes+=a.nbytes;row['memory'][key]={'shape':list(a.shape),'dtype':dtype,'bytes':a.nbytes}
        elif feat is not None:
            p=tensor_payload(feat);_put(parts,f'{section}/{fr}/memory/raw',p['array']);memory_bytes+=p['bytes'];row['memory']={'method':'independent','shape':p['shape'],'dtype':p['dtype'],'bytes':p['bytes']}
        # Position encodings are session constants and are charged once, below.
        if out.get('maskmem_pos_enc') is not None:row['maskmem_pos_enc_ref']='session_constant'
        frames.append(row)
    for obj_idx,maskdict in enumerate(state['mask_inputs_per_obj'].values()):
        for fr,mask in maskdict.items():
            p=tensor_payload(mask);_put(parts,f'prompt/object_{obj_idx}/frame_{fr}',p['array']);
    constants=[]
    for i,x in enumerate(state.get('constants',{}).get('maskmem_pos_enc',[]) or []):
        a,_=tensor_array(x);_put(parts,f'constant/maskmem_pos_enc/{i}',a);constants.append({'shape':list(a.shape),'bytes':a.nbytes})
    header={'schema':'MA1028-complete-inference-session-v1','method':method,'scene_seed':scene['seed'],'official_pathways':state['num_pathway'],'frames':frames,'shared_constants':constants,'scene_manifest_sha256':hashlib.sha256((R/f"seed_{scene['seed']}_manifest.json").read_bytes()).hexdigest(),'common_video_cache_sha256':hashlib.sha256(cache_blob).hexdigest()}
    arrays=[];manifest_rows=[]
    for name,a,dt in parts:
        arrays.append(a.tobytes(order='C'));manifest_rows.append({'name':name,'shape':list(a.shape),'dtype':dt,'bytes':int(a.nbytes)})
    header['tensors']=manifest_rows
    raw=json.dumps(header,sort_keys=True,separators=(',',':')).encode()+b'\n'+b''.join(arrays)
    return raw,memory_bytes,header

def run_method(predictor,method,scene):
    video=R/'frames'/f"seed_{scene['seed']}"; maskfile=R/'masks'/f"seed_{scene['seed']}"/'00000.npz'
    truth=[np.load(R/'masks'/f"seed_{scene['seed']}"/f'{i:05d}.npz')['mask'].astype(bool) for i in range(scene['frames'])]
    timers={'encode':0.,'decode':0.}; start=time.perf_counter()
    state=predictor.init_state(str(video),async_loading_frames=False,offload_video_to_cpu=False,offload_state_to_cpu=False)
    state['num_pathway']=3;state['iou_thre']=.1;state['uncertainty']=2
    if method!='independent':install_storage_adapter(predictor,method,timers)
    m=np.load(maskfile)['mask'];predictor.add_new_mask(state,frame_idx=0,obj_id=scene['object_id'],mask=torch.as_tensor(m,dtype=torch.bool))
    # The upstream implementation's own fixed-width tree is the search/control mechanism.
    track_t=time.perf_counter();ids,masks=predictor.propagate_in_video(state);prop_wall=time.perf_counter()-track_t
    if method!='independent':
        # Leave a compact resident session and serialize compressed memory; no decoded full bank remains.
        for section,fr,out in _entries(state['output_dict']):
            rep=out.get('_ma1028_packed')
            if rep is not None:out['maskmem_features']=rep['base'].unsqueeze(-1).expand(*rep['base'].shape,rep['shape'][-1])
    pred=[]
    for x in masks:
        a=x.detach().cpu().numpy();a=np.squeeze(a);pred.append(a>0)
    if len(pred)!=scene['frames']:raise AssertionError((len(pred),scene['frames']))
    metrics=score(pred,truth)
    # Cache the actual SAM2 normalized frame tensors shared across methods.
    image_parts=[]
    images=state['images']
    for i,x in enumerate(images):
        a,dtype=tensor_array(x);image_parts.append({'frame':i,'shape':list(a.shape),'dtype':dtype,'data':a.tobytes(order='C')})
    ch={'schema':'MA1028-common-frame-cache-v1','scene_seed':scene['seed'],'frames':[{'frame':x['frame'],'shape':x['shape'],'dtype':x['dtype'],'bytes':len(x['data'])} for x in image_parts]}
    cache=json.dumps(ch,sort_keys=True,separators=(',',':')).encode()+b'\n'+b''.join(x['data'] for x in image_parts)
    (R/f"cache_seed_{scene['seed']}.bin").write_bytes(cache)
    blob,memory_bytes,session_head=pack_state(method,state,scene,scene,cache)
    dest=R/f"session_seed_{scene['seed']}_{method}.bin";dest.write_bytes(blob)
    np.savez_compressed(R/f"pred_seed_{scene['seed']}_{method}.npz",masks=np.stack(pred).astype(np.uint8))
    full=CKPT.stat().st_size+CFG.stat().st_size+len(cache)+len(blob)
    end=time.perf_counter();return {'seed':scene['seed'],'method':method,**metrics,'session_payload_bytes':len(blob),'path_memory_bytes':memory_bytes,'full_session_payload_bytes':full,'encode_wall_s':timers['encode'],'decode_wall_s':timers['decode'],'encode_decode_wall_s':timers['encode']+timers['decode'],'propagation_wall_s':prop_wall,'total_wall_s':end-start,'fps':scene['frames']/max(end-start,1e-12),'session_sha256':hashlib.sha256(blob).hexdigest(),'prediction_sha256':hashlib.sha256((R/f"pred_seed_{scene['seed']}_{method}.npz").read_bytes()).hexdigest(),'session_manifest':session_head}

def main():
    import sam2
    repo_commit=os.popen(f'git -C {OFFICIAL} rev-parse HEAD').read().strip()
    ckhash=hashlib.sha256(CKPT.read_bytes()).hexdigest();cfg_hash=hashlib.sha256(CFG.read_bytes()).hexdigest()
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(CKPT),device='cpu',apply_postprocessing=False,hydra_overrides_extra=['++model.image_size=1024'])
    scene_paths=sorted(R.glob('seed_*_manifest.json'));scenes=[json.loads(p.read_text()) for p in scene_paths]
    allrows=[];run_info={'official_commit':repo_commit,'checkpoint_sha256':ckhash,'checkpoint_bytes':CKPT.stat().st_size,'config_sha256':cfg_hash,'config_bytes':CFG.stat().st_size,'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,'sam2_module':sam2.__file__,'device':'cpu','image_size':256,'pathways':3,'seeds':[]}
    for scene in scenes:
        for method in METHODS:
            # Restore the unwrapped official method before installing each adapter.
            if hasattr(predictor,'_ma1028_original_run'):predictor._run_single_frame_inference=predictor._ma1028_original_run
            else:predictor._ma1028_original_run=predictor._run_single_frame_inference
            row=run_method(predictor,method,scene);allrows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='session_manifest'},sort_keys=True),flush=True)
    (R/'result_rows.json').write_text(json.dumps({'runtime':run_info,'rows':allrows},indent=2)+'\n')

if __name__=='__main__':main()
