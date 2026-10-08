import argparse,io,json,time,struct
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
T,H,DX=32,32,16
METHODS=['tied','binary_mask','continuous_gate','mirror','independent']

def rotate(v,theta):
    out=[]; omega=torch.arange(1,H//2+1,dtype=v.dtype,device=v.device)
    for p in range(H//2):
        a=theta*omega[p];c,s=torch.cos(a),torch.sin(a)
        x,y=v[2*p],v[2*p+1];out.extend([c*x-s*y,s*x+c*y])
    return torch.stack(out)

def rotate_all(v,theta):
    omega=torch.arange(1,H//2+1,dtype=v.dtype,device=v.device)[None,:]
    a=theta[:,None]*omega;c,s=torch.cos(a),torch.sin(a)
    x=v[0::2][None,:];y=v[1::2][None,:]
    pairs=torch.stack([c*x-s*y,s*x+c*y],dim=-1)
    return pairs.reshape(len(theta),H)

def world(seed):
    g=torch.Generator().manual_seed(seed)
    enc=torch.randn(DX,H,generator=g)/DX**.5
    teacher=torch.randn(H,generator=g)/H**.5
    angles=torch.rand(T,generator=g)*.30-.15
    def draw(n,offset):
        gg=torch.Generator().manual_seed(seed+offset);x=torch.randn(n,DX,generator=gg);h=torch.relu(x@enc)
        y=torch.stack([h@rotate(teacher,a) for a in angles]);return h,y
    h,y=draw(2048,101);hv,yv=draw(1024,202);ht,yt=draw(2048,303)
    return enc,h,y,hv,yv,ht,yt

class Model(torch.nn.Module):
    def __init__(self,kind,seed):
        super().__init__();self.kind=kind;g=torch.Generator().manual_seed(seed)
        if kind in ('tied','mirror','binary_mask','continuous_gate'):self.w=torch.nn.Parameter(torch.randn(H,generator=g)*.02)
        if kind=='binary_mask':self.logits=torch.nn.Parameter(torch.zeros(T,H))
        if kind=='continuous_gate':self.gate=torch.nn.Parameter(torch.ones(T,H))
        if kind=='mirror':self.theta=torch.nn.Parameter(torch.zeros(T))
        if kind=='independent':self.w=torch.nn.Parameter(torch.randn(T,H,generator=g)*.02)
    def vectors(self):
        if self.kind=='tied':return self.w.expand(T,-1)
        if self.kind=='independent':return self.w
        if self.kind=='continuous_gate':return self.w[None,:]*self.gate
        if self.kind=='binary_mask':
            probs=torch.sigmoid(self.logits);hard=(probs>=.5).to(probs.dtype);gate=hard+probs-probs.detach();return self.w[None,:]*gate
        return rotate_all(self.w,self.theta)

def serialize(m,enc_seed):
    if m.kind=='binary_mask':
        masks=(m.logits.detach().cpu().numpy()>=0).astype(np.uint8);packed=np.packbits(masks,axis=1,bitorder='little')
        arrays=[m.w.detach().cpu().numpy().astype('<f4').tobytes(),packed.tobytes()]
        code='packed_binary_mask'
    elif m.kind=='continuous_gate':
        arrays=[m.w.detach().cpu().numpy().astype('<f4').tobytes(),m.gate.detach().cpu().numpy().astype('<f2').tobytes()];code='fp16_continuous_gates'
    elif m.kind=='mirror':
        scale=0.2/127;vals=torch.clamp(torch.round(m.theta.detach().cpu()/scale),-127,127).to(torch.int8).numpy()
        arrays=[m.w.detach().cpu().numpy().astype('<f4').tobytes(),vals.tobytes()];code='int8_angle_scale_0.2_over_127'
    elif m.kind=='tied':arrays=[m.w.detach().cpu().numpy().astype('<f4').tobytes()];code='shared_readout'
    else:arrays=[m.w.detach().cpu().numpy().astype('<f4').tobytes()];code='independent_readouts'
    cfg={'method':m.kind,'tasks':T,'hidden':H,'input':DX,'encoder_seed':enc_seed,'activation':'relu','view':'pairwise_givens_omega_1_to_16','code':code}
    head=json.dumps(cfg,separators=(',',':'),sort_keys=True).encode();return struct.pack('<H',len(head))+head+b''.join(arrays)

def unpack_payload(blob):
    n=struct.unpack('<H',blob[:2])[0];cfg=json.loads(blob[2:2+n]);buf=memoryview(blob)[2+n:];k=cfg['method'];offset=0
    def array(dtype,count):
        nonlocal offset
        a=np.frombuffer(buf,dtype=dtype,count=count,offset=offset).copy();offset+=a.nbytes;return a
    if k=='tied':return torch.from_numpy(array('<f4',H)).expand(T,-1)
    if k=='independent':return torch.from_numpy(array('<f4',T*H).reshape(T,H))
    if k=='binary_mask':
        w=torch.from_numpy(array('<f4',H));packed=array('u1',T*((H+7)//8)).reshape(T,-1);bits=np.unpackbits(packed,axis=1,bitorder='little')[:,:H];return w[None,:]*torch.from_numpy(bits.copy()).float()
    w=torch.from_numpy(array('<f4',H))
    if k=='continuous_gate':return w[None,:]*torch.from_numpy(array('<f2',T*H).reshape(T,H)).float()
    codes=torch.from_numpy(array('i1',T)).float();theta=codes*(0.2/127);return rotate_all(w,theta)

def run(seed,lr,updates,split):
    enc,h,y,hv,yv,ht,yt=world(seed);rows=[]
    for idx,k in enumerate(METHODS):
        m=Model(k,seed*100+idx);opt=torch.optim.Adam(m.parameters(),lr=lr);start=time.time();seen=0
        for step in range(updates):
            ix=(torch.arange(128)+step*128)%len(h);pred=h[ix]@m.vectors().T;loss=(pred-y[:,ix].T).square().mean();opt.zero_grad();loss.backward();opt.step();seen+=len(ix)*T
        train_s=time.time()-start;blob=serialize(m,seed);vectors=unpack_payload(blob)
        with torch.no_grad():
            pred=ht@vectors.T; den=yt.square().mean().item();mse=(pred-yt.T).square().mean().item()/den
            t0=time.time()
            for _ in range(10): ht@vectors.T
            throughput=10*len(ht)*T/(time.time()-t0)
        mac=updates*128*T*H*3
        rows.append({'condition':split,'world_or_seed':seed,'method':k,'serialized_bytes':len(blob),'train_tokens_or_examples':seen,'optimizer_updates':updates,'active_compute_proxy':mac,'wall_time_s':train_s,'primary_metric':'normalized_heldout_mse','primary_value':mse,'secondary_metric':'inference_examples_per_s','secondary_value':throughput,'status_note':f'lr={lr}; payload_reconstruction_evaluated'})
    return rows

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--world',type=int,required=True);p.add_argument('--lr',type=float,required=True);p.add_argument('--updates',type=int,default=600);p.add_argument('--split',default='development');p.add_argument('--out',required=True);a=p.parse_args();r=run(a.world,a.lr,a.updates,a.split);Path(a.out).write_text(json.dumps(r,indent=2));
 for x in r:print(x['world_or_seed'],x['method'],f"nMSE={x['primary_value']:.5g}",f"bytes={x['serialized_bytes']}",f"s={x['wall_time_s']:.2f}")
