"""Verify actual payload bytes and replay held-out development metrics from them."""
import hashlib,json,sys
from pathlib import Path
import torch
from model import (TOKENS,TOKEN_DIM,CoordinateDecoder,SharedFactorGenerator,all_coords,
                   decode_y4m,evaluate,split_indices)
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'source';DATA=Path('/tmp/ma945-data')
raw=json.loads((SOURCE/'development_raw.json').read_text())['rows']
cache={};clips={};max_delta=0.0
TOLERANCE=1e-5
for x in raw:
 p=ROOT/x['payload_artifact'];blob=p.read_bytes();digest=hashlib.sha256(blob).hexdigest()
 if len(blob)!=x['library_payload_bytes'] or digest!=x['library_payload_sha256']:
  raise SystemExit(f'payload bytes/hash mismatch: {p}')
 if p not in cache:
  obj=torch.load(p,map_location='cpu',weights_only=True)
  dec=CoordinateDecoder();dec.load_state_dict(obj['decoder_state']);dec.eval()
  cache[p]=(obj,dec)
 clip=x['clip']
 if clip not in clips:clips[clip]=decode_y4m(DATA/f'{clip}.y4m')
 obj,dec=cache[p];ids=obj['metadata']['video_ids'];idx=ids.index(clip);method=x['method'];rank=x['rank'];rep=obj['representation']
 if method=='native_full':tokens=rep['video_tokens'][idx]
 elif method=='lowrank':tokens=rep['mean_token_table']+torch.einsum('r,rnd->nd',rep['video_codes'][idx],rep['shared_token_basis'])
 else:
  gm='mirror' if method=='mirror_private' else method
  gen=SharedFactorGenerator(gm,rank,4);gen.load_state_dict(rep['shared_factor_generator'],strict=False);gen.eval()
  with torch.inference_mode():tokens=gen.table_from_code(rep['video_codes'][idx])
  if method=='mirror_private':tokens=tokens+rep['private_slot_codes'][idx]@rep['private_token_basis']
 coords,targets=all_coords(clips[clip]);_,eval_idx=split_indices(len(coords),x['seed'])
 mask=torch.zeros(len(coords),dtype=torch.bool);mask[eval_idx]=True
 got=evaluate(dec,tokens,clips[clip],mask)
 for key in ('heldout_mse','heldout_psnr_db','heldout_ssim'):
  delta=abs(got[key]-x[key]);max_delta=max(max_delta,delta)
  if delta>TOLERANCE:raise SystemExit(f"{clip} {method} r{rank} {key}: {got[key]} vs {x[key]} (delta {delta})")
 if got['eval_coordinates']!=x['eval_coordinates']:raise SystemExit('eval coordinate count mismatch')
rep={'checked':True,'rows':len(raw),'payload_files':len(cache),'fields':['actual payload size/hash','heldout_mse','heldout_psnr_db','heldout_ssim','eval_coordinates'],'max_abs_metric_delta':max_delta,'absolute_tolerance':TOLERANCE,'fresh_accessed':False}
(SOURCE/'metric_replay.json').write_text(json.dumps(rep,indent=2)+'\n')
print(json.dumps(rep,indent=2))
