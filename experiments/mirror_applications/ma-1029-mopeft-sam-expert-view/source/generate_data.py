"""Generate deterministic visual-domain segmentation data after protocol freeze."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

E=Path(__file__).resolve().parents[1]; R=E/'runs'; IMG=R/'images'; MASK=R/'masks'
SEEDS={'dev':102900,'audit_1':102901,'audit_2':102902,'audit_3':102903}
SPLITS={'train':2,'dev':2,'audit':4}; DOMAINS=('texture_contrast','low_contrast_camouflage','partial_occlusion')
N=256; yy,xx=np.mgrid[:N,:N]

def draw_example(rng,domain,index):
    phase=float(rng.uniform(0,2*np.pi)); noise=rng.normal(0,5,(N,N,1))
    if domain=='texture_contrast':
        base=np.stack([118+16*np.sin(xx/13+phase),125+15*np.cos(yy/17+phase),132+13*np.sin((xx+yy)/21)],-1)
        color=np.array([236,48,38])
    elif domain=='low_contrast_camouflage':
        base=np.stack([140+12*np.sin(xx/9+phase),128+11*np.cos(yy/11+phase),116+8*np.sin((xx+yy)/13)],-1)
        color=np.array([153,137,124])
    else:
        base=np.stack([110+18*np.sin(xx/16+phase),119+15*np.cos(yy/14+phase),132+17*np.sin((xx+yy)/24)],-1)
        color=np.array([240,184,32])
    canvas=np.clip(base+noise,0,255).astype(np.uint8)
    cx=int(rng.integers(78,178));cy=int(rng.integers(78,178));rad=int(rng.integers(25,39));kind=index%3
    mask_im=Image.new('L',(N,N));md=ImageDraw.Draw(mask_im)
    if kind==0:md.ellipse((cx-rad,cy-rad,cx+rad,cy+rad),fill=1)
    elif kind==1:md.rectangle((cx-rad,cy-rad,cx+rad,cy+rad),fill=1)
    else:md.polygon([(cx,cy-rad-4),(cx-rad,cy+rad),(cx+rad,cy+rad)],fill=1)
    shape=np.asarray(mask_im)>0
    # A deterministic spatial pattern helps distinguish the target from a color-only cue.
    stripe=((xx+2*yy+index*7)%19)<8
    texture=np.where(stripe[...,None],np.clip(color+24,0,255),np.clip(color-19,0,255)).astype(np.uint8)
    canvas[shape]=texture[shape]
    if domain=='partial_occlusion':
        occ=((xx+yy+index*13)>int(2*cx+cy-18)) & ((xx+yy+index*13)<int(2*cx+cy+17))
        visible=shape & ~occ
        canvas[occ]=np.clip(np.stack([77+xx[occ]%27,83+yy[occ]%31,96+(xx[occ]+yy[occ])%35],-1),0,255).astype(np.uint8)
    else:visible=shape
    ys,xs=np.where(visible)
    if len(xs)==0:raise RuntimeError('generated empty visible mask')
    margin=8;box=[max(0,int(xs.min())-margin),max(0,int(ys.min())-margin),min(N-1,int(xs.max())+margin),min(N-1,int(ys.max())+margin)]
    return Image.fromarray(canvas,'RGB'),visible.astype(np.uint8),box

parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['dev','audit'],required=True);phase=parser.parse_args().phase
rows=[]
for split,n in SPLITS.items():
  if phase=='dev' and split not in ('train','dev'):continue
  if phase=='audit' and split not in ('train','audit'):continue
  for domain_i,domain in enumerate(DOMAINS):
    for world,seed in SEEDS.items():
      if phase=='dev' and world!='dev':continue
      if phase=='audit' and not world.startswith('audit_'):continue
      folder=f'{world}/{split}/{domain}';(IMG/folder).mkdir(parents=True,exist_ok=True);(MASK/folder).mkdir(parents=True,exist_ok=True)
      rng=np.random.default_rng(seed+domain_i*1009+(0 if split=='train' else 100 if split=='dev' else 200))
      for i in range(n):
        im,mask,box=draw_example(rng,domain,i);name=f'{i:03d}'
        ip=IMG/folder/f'{name}.jpg';mp=MASK/folder/f'{name}.npz';im.save(ip,quality=97,subsampling=0);np.savez_compressed(mp,mask=mask,box=np.asarray(box,dtype=np.float32))
        rows.append({'world':world,'world_seed':seed,'split':split,'domain':domain,'index':i,'image':str(ip.relative_to(R)),'mask':str(mp.relative_to(R)),'box':box,'image_sha256':hashlib.sha256(ip.read_bytes()).hexdigest(),'mask_sha256':hashlib.sha256(mp.read_bytes()).hexdigest()})
manifest={'schema':'MA1029-synthetic-sam-feature-adapter-v1','phase':phase,'size':[N,N],'domains':DOMAINS,'seeds':SEEDS,'counts':SPLITS,'rows':rows}
(R/f'data_manifest_{phase}.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'phase':phase,'images':len(rows)},sort_keys=True))
