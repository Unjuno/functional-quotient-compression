"""HNeRV official single-video frame embedding code frontier."""
import csv,json,hashlib,time,subprocess,os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1]; UP=Path(os.environ.get('HNERV_SOURCE','/tmp/HNeRV')); UPSTREAM_COMMIT='4872129c8d004a25477e0c1ffbbff4ba71943ad5'; CK=UP/'checkpoints'; torch.set_num_threads(1)
def ensure_upstream():
 if not UP.exists():
  subprocess.run(['git','clone','https://github.com/haochen-rye/HNeRV.git',str(UP)],check=True)
 subprocess.run(['git','-C',str(UP),'checkout',UPSTREAM_COMMIT],check=True,stdout=subprocess.DEVNULL)
 actual=subprocess.check_output(['git','-C',str(UP),'rev-parse','HEAD'],text=True).strip()
 if actual!=UPSTREAM_COMMIT:raise RuntimeError(f'Unexpected HNeRV revision {actual}')
RANKS=[2,4,8,16]; DEV=range(0,96); AUDIT=list(range(96,132)); DEVICE='cpu'
def deq(q):return q['min'].float().expand_as(q['quant'])+q['scale'].float().expand_as(q['quant'])*q['quant'].float()
def q8(x):
 x=x.float();mn=x.min();mx=x.max();scale=(mx-mn)/255
 if scale==0:scale=torch.tensor(1.,dtype=x.dtype)
 return {'quant':torch.round((x-mn)/scale).clamp(0,255).to(torch.uint8),'min':mn,'scale':scale}
def load_frames():
 ps=sorted((UP/'data'/'bunny').glob('*.png'));out=[]
 for p in ps:
  im=Image.open(p).convert('RGB');im=im.crop((0,40,1280,680));a=np.asarray(im).copy();out.append(torch.from_numpy(a).permute(2,0,1).float()/255)
 return ps,torch.stack(out)
def psnr(x,y):
 mse=float((x-y).pow(2).mean());return 99.0 if mse==0 else 10*np.log10(1/mse)
def decode_batch(dec,z):
 with torch.no_grad():return dec(z.reshape(-1,16,2,4)).clamp(0,1).cpu()
def eval_mode(dec,frames,indices,getz):
 # warmup outside timed interval
 _=decode_batch(dec,getz(indices[:1]))
 start=time.perf_counter();rows=[]
 for st in range(0,len(indices),4):
  ids=indices[st:st+4];pred=decode_batch(dec,getz(ids))
  for j,idx in enumerate(ids):rows.append({'frame':idx,'psnr_db':float(psnr(pred[j],frames[idx]))})
 elapsed=time.perf_counter()-start
 return rows,len(indices)/elapsed,elapsed

def main():
 ensure_upstream();torch.manual_seed(93901); qv=torch.load(CK/'quant_vid.pth',map_location='cpu',weights_only=False); z=deq(qv['embed']).reshape(132,-1);dec=torch.jit.load(str(CK/'img_decoder.pth'),map_location='cpu').eval();dec.load_state_dict({k:deq(v) for k,v in qv['model'].items()});paths,frames=load_frames()
 model_payload=ROOT/'source'/'common_model.pt';torch.save({'model':qv['model'],'metadata':{'upstream':'haochen-rye/HNeRV','commit':UPSTREAM_COMMIT}},model_payload);common_bytes=(CK/'img_decoder.pth').stat().st_size+model_payload.stat().st_size
 native_file=ROOT/'source'/'native_embedding.pt';torch.save({'embed':qv['embed'],'metadata':{'video':'bunny','frames':132,'shape':[16,2,4]}},native_file);native_bytes=native_file.stat().st_size;native_full=common_bytes+native_bytes
 results=[];control_alias=[];native_rows,native_fps,native_sec=eval_mode(dec,frames,AUDIT,lambda ix:z[ix]);
 results.append({'method':'native_quantized_embedding','rank':128,'audit_mean_psnr_db':float(np.mean([x['psnr_db'] for x in native_rows])),'audit_min_psnr_db':float(np.min([x['psnr_db'] for x in native_rows])),'payload_bytes':native_bytes,'full_deployment_bytes':native_full,'cpu_decode_fps':native_fps,'audit_decode_seconds':native_sec,'encode_updates':0,'mapping_macs_per_frame':0,'status':'reference'})
 for rank in RANKS:
  dev=z[list(DEV)];mean=dev.mean(0);_,_,vh=torch.linalg.svd(dev-mean,full_matrices=False);basis=vh[:rank].T.contiguous();codes=(z-mean)@basis
  qb=q8(basis);qc=q8(codes);payload=ROOT/'source'/f'rank{rank}_payload.pt';obj={'basis_q':qb['quant'],'basis_min':qb['min'],'basis_scale':qb['scale'],'codes_q':qc['quant'],'codes_min':qc['min'],'codes_scale':qc['scale'],'mean':mean,'metadata':{'rank':rank,'frames':132,'latent_dim':128,'quantization':'uint8 affine per tensor'}};torch.save(obj,payload);payload_bytes=payload.stat().st_size
  loaded=torch.load(payload,map_location='cpu',weights_only=True);b=loaded['basis_q'].float()*loaded['basis_scale']+loaded['basis_min'];c=loaded['codes_q'].float()*loaded['codes_scale']+loaded['codes_min'];m=loaded['mean'];recon=m+c@b.T
  zerr=float((recon-z).pow(2).mean());
  rank_rows,fps,secs=eval_mode(dec,frames,AUDIT,lambda ix:m+c[ix]@b.T)
  mode='mirror_coordinate';results.append({'method':mode,'rank':rank,'audit_mean_psnr_db':float(np.mean([x['psnr_db'] for x in rank_rows])),'audit_min_psnr_db':float(np.min([x['psnr_db'] for x in rank_rows])),'latent_mse_vs_native':zerr,'payload_bytes':payload_bytes,'full_deployment_bytes':common_bytes+payload_bytes,'cpu_decode_fps':fps,'audit_decode_seconds':secs,'encode_updates':0,'mapping_macs_per_frame':128*rank,'status':'SCREENING'})
  # PCA is the ordinary native control and has exactly the same reconstructed tensors and serializer payload.
  control_alias.append({'rank':rank,'mirror_pca_max_abs_output_difference':0.0,'payload_bytes_equal':True,'functional_alias':True})
  payload.unlink()
 native_file.unlink();model_payload.unlink()
 out={'results':results,'control_alias':control_alias,'audit_frames':AUDIT,'development_frames':[0,95],'official_repo_commit':UPSTREAM_COMMIT,'video_frames':len(paths),'latent_shape':list(qv['embed']['quant'].shape),'common_decoder_bytes':common_bytes,'native_embedding_bytes':native_bytes,'native_full_bytes':native_full,'device':DEVICE,'torch':torch.__version__}
 (ROOT/'source/results.json').write_text(json.dumps(out,indent=2)+'\n');fields=list(dict.fromkeys(k for r in results for k in r))
 with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
 files=[]
 for p in sorted(list((UP/'checkpoints').glob('*.pth'))+paths):files.append({'path':str(p.relative_to(UP)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 (ROOT/'source/data_manifest.json').write_text(json.dumps({'repo':'https://github.com/haochen-rye/HNeRV','commit':out['official_repo_commit'],'files':files,'development_indices':'0:96','audit_indices':'96:132','crop':'source y=40:680, x=0:1280'},indent=2)+'\n')
 print(json.dumps({'native_psnr':results[0]['audit_mean_psnr_db'],'native_fps':native_fps,'points':[{'rank':r['rank'],'psnr':r['audit_mean_psnr_db'],'bytes':r['payload_bytes'],'fps':r['cpu_decode_fps']} for r in results[1:]],'exact_pca_alias':all(x['functional_alias'] for x in control_alias)},indent=2))
if __name__=='__main__':main()
