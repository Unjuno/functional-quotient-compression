"""Interactive correction replay and compact polar-code screen using official SAM2."""
import argparse,hashlib,itertools,json,io,os,pathlib,time
os.environ.setdefault('SAM2_BUILD_CUDA','0')
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
E=pathlib.Path(__file__).resolve().parents[1];RUNS=E/'runs';ROOT=RUNS/'worlds';REPO=pathlib.Path('/tmp/ma1027-sam2');CKPT=REPO/'sam2.1_hiera_tiny.pt';CONFIG=REPO/'sam2/configs/sam2.1/sam2.1_hiera_t.yaml'
IDS=[11,37,42,99];METHODS=['no_correction','direct_clicks','grid8_clicks','mirror_polar','full_mask_upper'];torch.set_num_threads(4)

def serialize_arrays(header,arrays):
    manifest=[];chunks=[]
    for name,a in arrays.items():
        a=np.ascontiguousarray(a);manifest.append({'name':name,'shape':list(a.shape),'dtype':a.dtype.str,'bytes':a.nbytes});chunks.append(a.tobytes())
    header['tensors']=manifest;return json.dumps(header,sort_keys=True,separators=(',',':')).encode()+b'\n'+b''.join(chunks),manifest

def decode_polar(center,theta,radii):
    center=np.asarray(center,dtype=np.float32);theta=np.asarray(theta,dtype=np.float32);r=np.asarray(radii,dtype=np.float32)
    d=np.stack([np.cos(theta),np.sin(theta)],axis=-1)*r[...,None]
    p=np.clip(center+d,0,255);n=np.clip(center-d,0,255)
    return np.stack([p,n],axis=1)

def click_history_blob(method,events,radii=None):
    frames=[e['frame'] for e in events];pairs=np.asarray([e['pairs'] for e in events],dtype=np.float32)
    header={'schema':'MA1031-correction-history-v1','method':method,'frames':frames,'object_ids':IDS,'click_order':['positive','negative'],'labels_reconstructible':[1,0]}
    if method=='direct_clicks':arrays={'xy_float32':pairs.astype('<f4')}
    elif method=='grid8_clicks':arrays={'xy_grid8_uint8':np.rint(pairs/8).clip(0,31).astype('u1')}
    elif method=='mirror_polar':
        centers=(pairs[:,:,0,:]+pairs[:,:,1,:])/2;delta=pairs[:,:,0,:]-pairs[:,:,1,:];angles=np.arctan2(delta[...,1],delta[...,0])
        arrays={'shared_half_radius_float16':np.asarray(radii,dtype='<f2'),'frame_midpoint_float16':centers.astype('<f2'),'view_angle_float16':angles.astype('<f2')}
    else:arrays={}
    return serialize_arrays(header,arrays)[0]

def decode_grid8(payload):
    # The runtime decoder for this fixed-grid control is deterministic from the codebook rule.
    return np.asarray(payload,dtype=np.float32)*8

def point_pair(pred,truth):
    fn=np.asarray(truth,bool)&~np.asarray(pred,bool);fp=np.asarray(pred,bool)&~np.asarray(truth,bool)
    ys,xs=np.where(fn if fn.any() else truth)
    if len(xs)==0:raise ValueError('object mask is empty')
    pos=np.array([float(np.median(xs)),float(np.median(ys))],dtype=np.float32)
    if fp.any():
        ys,xs=np.where(fp);neg=np.array([float(np.median(xs)),float(np.median(ys))],dtype=np.float32)
    else:
        y0,x0=np.where(truth);cx=float(np.median(x0));cy=float(np.median(y0));neg=np.array([min(255.,max(x0)+8.),cy],dtype=np.float32)
        if truth[int(round(neg[1])),int(round(neg[0]))]:neg=np.array([max(0.,min(x0)-8.),cy],dtype=np.float32)
    return np.stack([pos,neg])

def iou(a,b):
    inter=np.logical_and(a,b).sum();union=np.logical_or(a,b).sum();return float(inter/union) if union else 1.
def boundary_f(pred,truth,tol=2):
    import torch.nn.functional as F
    p=torch.as_tensor(pred,dtype=torch.float32)[None,None];g=torch.as_tensor(truth,dtype=torch.float32)[None,None]
    pb=(p-(F.avg_pool2d(p,3,1,1)==1).float())>0;gb=(g-(F.avg_pool2d(g,3,1,1)==1).float())>0
    k=2*tol+1;pd=F.max_pool2d(pb.float(),k,1,tol)>0;gd=F.max_pool2d(gb.float(),k,1,tol)>0
    pr=float((pb&gd).sum()/(pb.sum()+1e-8));rc=float((gb&pd).sum()/(gb.sum()+1e-8));return 2*pr*rc/(pr+rc+1e-8)
def score_masks(preds,truths):
    js=[];fs=[];switch=0;prev=None
    for f in range(len(preds)):
        mat=np.array([[iou(p,g) for g in truths[f]] for p in preds[f]])
        assignment=max(itertools.permutations(range(len(mat))),key=lambda q:sum(mat[i,q[i]] for i in range(len(q))))
        if prev is not None:switch+=sum(assignment[i]!=prev[i] for i in range(len(assignment)))
        prev=assignment
    for f in range(1,len(preds)):
        for i in range(len(preds[f])):js.append(iou(preds[f][i],truths[f][i]));fs.append(boundary_f(preds[f][i],truths[f][i]))
    return {'mean_j':float(np.mean(js)),'mean_boundary_f':float(np.mean(fs)),'mean_jf':float((np.mean(js)+np.mean(fs))/2),'id_switches':int(switch)}

def run_one(world,method,radii=None):
    root=ROOT/world;scene=json.loads((root/'manifest.json').read_text());frames=root/'frames';masks=root/'masks'
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',str(CKPT),device='cpu',apply_postprocessing=False,hydra_overrides_extra=['++model.image_size=256'])
    start=time.perf_counter();state=predictor.init_state(str(frames),offload_video_to_cpu=False,offload_state_to_cpu=False)
    truths=[np.load(masks/f'{f:05}.npz')['masks'].astype(bool) for f in range(len(scene['rows']))]
    for oi,oid in enumerate(IDS):predictor.add_new_mask(state,0,oid,torch.from_numpy(truths[0][oi]))
    gen=predictor.propagate_in_video(state);preds=[];events=[];click_count=0;codes=[];mirror_codes=[]
    for frame,ids,logits in gen:
        current=(logits.detach().cpu().numpy()[:,0]>0)
        if frame==0:
            preds.append(current);continue
        raw=np.stack([point_pair(current[i],truths[frame][i]) for i in range(len(IDS))])
        event={'frame':int(frame),'pairs':raw.tolist()};events.append(event)
        corrected=logits
        if method=='direct_clicks':
            for i,oid in enumerate(IDS):_,_,corrected=predictor.add_new_points_or_box(state,frame,oid,points=raw[i],labels=[1,0],clear_old_points=True)
            click_count+=2*len(IDS);codes.append(raw)
        elif method=='grid8_clicks':
            quant=np.rint(raw/8).clip(0,31)*8
            for i,oid in enumerate(IDS):_,_,corrected=predictor.add_new_points_or_box(state,frame,oid,points=quant[i],labels=[1,0],clear_old_points=True)
            click_count+=2*len(IDS);codes.append(quant)
        elif method=='mirror_polar':
            center=(raw[:,0,:]+raw[:,1,:])/2;delta=raw[:,0,:]-raw[:,1,:];theta=np.arctan2(delta[...,1],delta[...,0]).astype('<f2').astype(np.float32);center=center.astype('<f2').astype(np.float32)
            encoded_center=center.astype('<f2');encoded_theta=theta.astype('<f2');decoded=decode_polar(encoded_center.astype(np.float32),encoded_theta.astype(np.float32),radii)
            for i,oid in enumerate(IDS):_,_,corrected=predictor.add_new_points_or_box(state,frame,oid,points=decoded[i],labels=[1,0],clear_old_points=True)
            click_count+=2*len(IDS);mirror_codes.append({'center':encoded_center,'theta':encoded_theta})
        elif method=='full_mask_upper':
            for i,oid in enumerate(IDS):_,_,corrected=predictor.add_new_mask(state,frame,oid,torch.from_numpy(truths[frame][i].astype(np.uint8)))
        elif method=='no_correction':
            pass
        else:raise ValueError(method)
        preds.append(corrected.detach().cpu().numpy()[:,0]>0)
    elapsed=time.perf_counter()-start;score=score_masks(preds,truths)
    if method in ('direct_clicks','grid8_clicks','mirror_polar'):
        event_pairs=np.stack(codes) if codes else None
        if method=='mirror_polar':
            centers=np.stack([c['center'] for c in mirror_codes]);angles=np.stack([c['theta'] for c in mirror_codes])
            blob,_=serialize_arrays({'schema':'MA1031-correction-history-v1','method':method,'frames':list(range(1,len(scene['rows']))),'object_ids':IDS,'labels_reconstructible':[1,0]}, {'shared_half_radius_float16':np.asarray(radii,dtype='<f2'),'frame_midpoint_float16':centers.astype('<f2'),'view_angle_float16':angles.astype('<f2')})
        elif method=='grid8_clicks':blob,_=serialize_arrays({'schema':'MA1031-correction-history-v1','method':method,'frames':list(range(1,len(scene['rows']))),'object_ids':IDS,'grid_px':8,'labels_reconstructible':[1,0]}, {'xy_grid8_uint8':np.rint(event_pairs/8).clip(0,31).astype('u1')})
        else:blob,_=serialize_arrays({'schema':'MA1031-correction-history-v1','method':method,'frames':list(range(1,len(scene['rows']))),'object_ids':IDS,'labels_reconstructible':[1,0]}, {'xy_float32':event_pairs.astype('<f4')})
    else:blob,_=serialize_arrays({'schema':'MA1031-correction-history-v1','method':method,'object_ids':IDS,'frames':[]}, {})
    # The model has already consumed these prompts and encoded their effect in SAM2 outputs.
    # Store history separately in its measured code; remove duplicate raw click dictionaries.
    bundle_state=dict(state);bundle_state['point_inputs_per_obj']={i:{} for i in range(len(IDS))}
    bio=io.BytesIO();torch.save({'sam2_inference_state':bundle_state,'correction_history':blob},bio);session=bio.getvalue()
    tag=f'{world}_{method}';h=RUNS/f'history_{tag}.bin';h.write_bytes(blob);predfile=RUNS/f'predictions_{tag}.npz';np.savez_compressed(predfile,masks=np.stack(preds).astype('u1'))
    source_bytes=sum(x.stat().st_size for x in pathlib.Path('/tmp/ma1027-sam2/sam2').rglob('*.py'))+sum((E/'source'/x).stat().st_size for x in ('run_interactive.py','generate_video.py'))+(E/'PROTOCOL.json').stat().st_size
    return {'world':world,'seed':scene['seed'],'method':method,**score,'click_count':click_count,'clicks_per_object':click_count//len(IDS),'click_history_bytes':len(blob),'session_state_bytes':len(session),'full_session_bytes':CKPT.stat().st_size+CONFIG.stat().st_size+source_bytes+len(session),'source_bytes':source_bytes,'fps':(len(scene['rows'])-1)/max(elapsed,1e-9),'wall_s':elapsed,'history_file':h.name,'history_sha256':hashlib.sha256(blob).hexdigest(),'session_sha256':hashlib.sha256(session).hexdigest(),'session_payload_retained':False,'prediction_file':predfile.name,'prediction_sha256':hashlib.sha256(predfile.read_bytes()).hexdigest(),'click_events':events if world=='dev' and method=='direct_clicks' else None,'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','audit'],required=True);args=ap.parse_args()
    if args.phase=='dev':
        pilot=run_one('dev','direct_clicks');clicks=np.asarray([e['pairs'] for e in pilot['click_events']]);radii=np.mean(np.linalg.norm(clicks[:,:,0,:]-clicks[:,:,1,:],axis=-1)/2,axis=0).astype('<f2')
        rows=[run_one('dev',m,radii) for m in METHODS]
        for r in rows:r.pop('click_events',None)
        (RUNS/'dev_rows.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
        freeze={'schema':'MA1031-dev-freeze-v1','base_commit':'d8ee28e09','dev_seed':103100,'shared_half_radius_float16':radii.tolist(),'radius_rule':'per-object mean half-distance of two direct dev corrections across frames 1-5','audit_seeds':[103101,103102,103103],'frozen':True}
        (RUNS/'frozen_settings.json').write_text(json.dumps(freeze,indent=2)+'\n');print(json.dumps({'freeze':freeze,'dev_rows':len(rows)},sort_keys=True))
    else:
        freeze=json.loads((RUNS/'frozen_settings.json').read_text());radii=np.asarray(freeze['shared_half_radius_float16'],dtype='<f2')
        rows=[]
        for world in ('audit_1','audit_2','audit_3'):
            ww=[run_one(world,m,radii) for m in METHODS];rows.extend(ww)
            with (RUNS/'audit_rows.jsonl').open('a') as f:
                for r in ww:r.pop('click_events',None);f.write(json.dumps(r,sort_keys=True)+'\n')
        print(json.dumps({'audit_rows':len(rows)},sort_keys=True))
if __name__=='__main__':main()
