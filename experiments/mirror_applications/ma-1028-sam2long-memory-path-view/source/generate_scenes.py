"""Deterministic single-object long clips with an occluder and reappearance."""
import hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

E=Path(__file__).resolve().parents[1]
FR=E/'runs'/'frames'; GT=E/'runs'/'masks'
SEEDS=(102801,102802,102803); N=256; T=12
FR.mkdir(parents=True,exist_ok=True); GT.mkdir(parents=True,exist_ok=True)
yy,xx=np.mgrid[:N,:N]

for seed in SEEDS:
    rng=np.random.default_rng(seed); scene=f'seed_{seed}'; (FR/scene).mkdir(exist_ok=True); (GT/scene).mkdir(exist_ok=True)
    rows=[]
    # Textured background is seed-specific; object moves behind a wide static occluder.
    phase=float(rng.uniform(0,6.28)); noise=rng.normal(0,1.0,(N,N,1))
    base=np.stack([110+13*np.sin(xx/15+phase)+6*np.cos(yy/27),112+10*np.sin(yy/17+phase),116+9*np.cos((xx+yy)/31+phase)],-1)
    base=np.clip(base+noise,0,255).astype(np.uint8)
    for t in range(T):
        cx=37+17*t; cy=int(124+13*np.sin(t*.48+phase))
        shape=((xx-cx)**2+(yy-cy)**2 <= 19**2)
        # Occlusion lasts multiple frames; the object reappears on the far side.
        occ=(xx>=105)&(xx<=174)&(yy>=34)&(yy<=222)
        visible=shape & ~occ
        canvas=base.copy()
        canvas[occ]=np.array([58,66,78],dtype=np.uint8)
        tex=(np.sin(xx/3)+np.cos(yy/4))>0
        canvas[shape & ~occ]=np.where(tex[shape & ~occ,None],np.array([235,72,40]),np.array([246,160,60]))
        im=Image.fromarray(canvas,'RGB'); jpg=f'{t:05d}.jpg'; im.save(FR/scene/jpg,quality=96,subsampling=0)
        np.savez_compressed(GT/scene/f'{t:05d}.npz',mask=visible.astype(np.uint8),frame=t,seed=seed)
        rows.append({'frame':t,'jpeg':jpg,'jpeg_sha256':hashlib.sha256((FR/scene/jpg).read_bytes()).hexdigest(),'mask_sha256':hashlib.sha256((GT/scene/f'{t:05d}.npz').read_bytes()).hexdigest(),'visible_pixels':int(visible.sum())})
    manifest={'schema':'MA1028-single-object-occlusion-v1','seed':seed,'frames':T,'size':[N,N],'object_id':seed%100000,'object_shape':'moving textured disk','occluder_box':[105,34,174,222],'rows':rows}
    (E/'runs'/f'{scene}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'seeds':SEEDS,'frames':T},sort_keys=True))
