"""Deterministic analytic multi-coil phantom generator for MA-1006 mechanism screens."""
import math
import numpy as np
import torch
import fastmri

def make_case(seed:int, scanner:str='A', size:int=64, coils:int=4, acceleration:int=4, center:int=8):
    rng=np.random.default_rng(seed); yy,xx=np.mgrid[-1:1:complex(size),-1:1:complex(size)]
    image=np.zeros((size,size),np.float32)
    for _ in range(7):
        cx,cy=rng.uniform(-.45,.45,2); a,b=rng.uniform(.10,.42),rng.uniform(.08,.35); ang=rng.uniform(0,math.pi)
        x=(xx-cx)*math.cos(ang)+(yy-cy)*math.sin(ang); y=-(xx-cx)*math.sin(ang)+(yy-cy)*math.cos(ang)
        image+=rng.uniform(-.3,1.0)*(((x/a)**2+(y/b)**2)<1)
    image=np.maximum(image,0); image/=max(float(image.max()),1e-6)
    phi=.08*np.sin(np.pi*xx)*np.cos(np.pi*yy)
    img=np.stack([image*np.cos(phi),image*np.sin(phi)],axis=-1)
    sens=[]
    shift=0.0 if scanner=='A' else .57
    for c in range(coils):
        ang=2*np.pi*c/coils+shift
        amp=1+.22*np.cos(np.pi*xx*np.cos(ang)+np.pi*yy*np.sin(ang)+shift)
        phase=.35*(xx*np.cos(ang)+yy*np.sin(ang)) + (0 if scanner=='A' else .65*np.sin(np.pi*yy+ang))
        sens.append(np.stack([amp*np.cos(phase),amp*np.sin(phase)],axis=-1))
    sens=np.asarray(sens,np.float32)
    denom=np.sqrt(np.sum(np.sum(sens*sens,axis=-1),axis=0))
    sens=sens/np.maximum(denom[None,:,:,None],1e-6)
    img_t=torch.from_numpy(img)[None,None]
    sens_t=torch.from_numpy(sens)[None]
    coil_img=fastmri.complex_mul(sens_t,img_t)
    kspace=fastmri.fft2c(coil_img)
    line=np.zeros(size,np.bool_);line[::acceleration]=True
    lo=size//2-center//2;line[lo:lo+center]=True
    mask=torch.from_numpy(line)[None,None,None,:,None]
    masked=kspace*mask
    target=torch.from_numpy(image)[None]
    return {'masked_kspace':masked.float(),'mask':mask,'target':target.float(),'full_kspace':kspace.float(),'num_low_frequencies':center,'scanner':scanner,'seed':seed}
