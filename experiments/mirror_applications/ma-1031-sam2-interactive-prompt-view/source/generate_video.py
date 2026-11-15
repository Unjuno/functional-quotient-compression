"""Generate deterministic crossing videos with a single causal appearance-drift event."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
E=Path(__file__).resolve().parents[1];P=E/'runs'/'worlds'
pa=argparse.ArgumentParser();pa.add_argument('--world',required=True);pa.add_argument('--seed',type=int,required=True);pa.add_argument('--phase',choices=['dev','audit'],required=True);a=pa.parse_args()
root=P/a.world;FR=root/'frames';GT=root/'masks';FR.mkdir(parents=True,exist_ok=True);GT.mkdir(parents=True,exist_ok=True)
N=256;T=6;IDS=[11,37,42,99];COLORS=[(226,40,35),(221,49,39),(215,55,42),(36,82,228)]
rng=np.random.default_rng(a.seed);yy,xx=np.mgrid[:N,:N]
bg=np.stack([110+8*np.sin(xx/17)+4*np.cos(yy/13),116+6*np.sin(yy/19),120+5*np.cos((xx+yy)/23)],-1);bg=np.clip(bg+rng.normal(0,.6,(N,N,1)),0,255).astype(np.uint8)
def masks_for(t):
 cx=[42+31*t,214-29*t,48+24*t,208-29*t];cy=[43+24*t,48+26*t,205-22*t,206-27*t];ms=[]
 ms.append((xx-cx[0])**2+(yy-cy[0])**2<=22**2)
 ms.append((np.abs(xx-cx[1])<=22)&(np.abs(yy-cy[1])<=22))
 im=Image.new('L',(N,N));ImageDraw.Draw(im).polygon([(cx[2],cy[2]-27),(cx[2]-25,cy[2]+22),(cx[2]+25,cy[2]+22)],fill=1);ms.append(np.asarray(im)>0)
 ms.append((((xx-cx[3])/28)**2+((yy-cy[3])/20)**2<=1))
 visible=[]
 for i,m in enumerate(ms):visible.append(m&~np.logical_or.reduce(ms[i+1:]) if i+1<len(ms) else m.copy())
 return ms,visible
rows=[]
for t in range(T):
 shapes,visible=masks_for(t);canvas=bg.copy();colors=COLORS.copy()
 # Object 99 alone changes appearance after a temporary partial occlusion begins.
 if t>=3:colors[3]=(35,222,85)
 for i,m in enumerate(shapes):canvas[m]=colors[i]
 # Deterministic narrow occluder crosses the rare object on frames 2-4.
 if t in (2,3,4):
  x0=100+13*t;occ=(xx>=x0)&(xx<x0+16)
  visible=[m&~occ for m in visible]
  canvas[:,x0:x0+16]=np.clip(bg[:,x0:x0+16].astype(np.int16)+[12,8,0],0,255).astype(np.uint8)
 im=Image.fromarray(canvas,'RGB');fp=FR/f'{t:05}.jpg';im.save(fp,quality=96,subsampling=0)
 np.savez_compressed(GT/f'{t:05}.npz',masks=np.stack(visible).astype(np.uint8),frame=t)
 rows.append({'frame':t,'jpeg':fp.name,'jpeg_sha256':hashlib.sha256(fp.read_bytes()).hexdigest(),'gt_sha256':hashlib.sha256((GT/f'{t:05}.npz').read_bytes()).hexdigest(),'visible_pixels':[int(m.sum()) for m in visible],'rare_object_color_changed':bool(t>=3)})
manifest={'schema':'MA1031-rare-appearance-interaction-v1','phase':a.phase,'world':a.world,'seed':a.seed,'frames':T,'size':[N,N],'object_ids':IDS,'rare_object_id':99,'appearance_drift_from_frame':3,'initial_prompt':'exact object masks on first frame','occlusion_frames':[2,3,4],'rows':rows}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'world':a.world,'seed':a.seed,'frames':T},sort_keys=True))
