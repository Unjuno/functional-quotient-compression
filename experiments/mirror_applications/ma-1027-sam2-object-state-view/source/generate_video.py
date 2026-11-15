"""Deterministic four-object crossing clip for the preregistered SAM2 screen."""
import hashlib,json,pathlib
import numpy as np
from PIL import Image,ImageDraw
E=pathlib.Path(__file__).resolve().parents[1];FR=E/'runs'/'video_frames';GT=E/'runs'/'gt_masks';FR.mkdir(parents=True,exist_ok=True);GT.mkdir(parents=True,exist_ok=True)
SEED=102701;N=256;T=5;IDS=[11,37,42,99];COLORS=[(230,38,34),(33,205,64),(37,83,232),(235,205,30)]
rng=np.random.default_rng(SEED);yy,xx=np.mgrid[:N,:N];bg=np.stack([112+8*np.sin(xx/17)+4*np.cos(yy/13),116+6*np.sin(yy/19),120+5*np.cos((xx+yy)/23)],-1);bg=np.clip(bg+rng.normal(0,.7,(N,N,1)),0,255).astype(np.uint8)
def masks_for(t):
 cx=[43+35*t,213-32*t,53+29*t,205-32*t];cy=[48+28*t,53+30*t,205-28*t,205-30*t];ms=[]
 ms.append(((xx-cx[0])**2+(yy-cy[0])**2<=22**2))
 ms.append((np.abs(xx-cx[1])<=22)&(np.abs(yy-cy[1])<=22))
 pts=np.array([[cx[2],cy[2]-27],[cx[2]-25,cy[2]+22],[cx[2]+25,cy[2]+22]],dtype=np.float32);im=Image.new('L',(N,N));ImageDraw.Draw(im).polygon([tuple(p) for p in pts],fill=1);ms.append(np.array(im)>0)
 ms.append((((xx-cx[3])/28)**2+((yy-cy[3])/20)**2<=1))
 # later draw order occludes earlier shapes; ground truth is visible area per object.
 visible=[]
 for i,m in enumerate(ms): visible.append(m & ~np.logical_or.reduce(ms[i+1:]) if i+1<len(ms) else m.copy())
 return ms,visible
rows=[]
for t in range(T):
 shapes,visible=masks_for(t);canvas=bg.copy()
 for i,m in enumerate(shapes):canvas[m]=COLORS[i]
 im=Image.fromarray(canvas,'RGB');path=FR/f'{t:05}.jpg';im.save(path,quality=96,subsampling=0)
 np.savez_compressed(GT/f'{t:05}.npz',masks=np.stack(visible).astype(np.uint8),frame=t)
 rows.append({'frame':t,'jpeg':path.name,'jpeg_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'gt_sha256':hashlib.sha256((GT/f'{t:05}.npz').read_bytes()).hexdigest(),'visible_pixels':[int(m.sum()) for m in visible]})
manifest={'schema':'MA1027-synthetic-crossing-v1','seed':SEED,'frames':T,'width':N,'height':N,'object_ids':IDS,'shapes':['circle','square','triangle','ellipse'],'colors_rgb':COLORS,'draw_order':IDS,'rows':rows}
(E/'runs'/'scene_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'seed':SEED,'frames':T,'rows':rows},sort_keys=True))
