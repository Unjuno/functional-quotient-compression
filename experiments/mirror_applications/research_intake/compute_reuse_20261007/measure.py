"""Replay random identity fixtures. No learned models or timing measurements."""
import csv,json,math,pathlib,platform,hashlib
import torch
from torch.nn import functional as F
from algebra import sign_explicit,sign_fused,mirror,attention,lazy_attention,partition_attention,mirror_mixture_fused
ROOT=pathlib.Path(__file__).parent
torch.set_num_threads(1)
rows=[]
def record(name,seed,dtype,left,right,expected='equal'):
    err=float((left-right).abs().max())
    scale=max(1.,float(left.abs().max()))
    tol=1e-9 if dtype==torch.float64 else 1e-4
    passed=err<=tol*scale if expected=='equal' else err>1e-3
    rows.append(dict(check=name,seed=seed,dtype=str(dtype),max_abs_error=err,scaled_error=err/scale,criterion=expected,pass_check=passed))
for dtype in (torch.float64,torch.float32):
    for seed in range(701,706):
        torch.manual_seed(seed)
        rnd=lambda *s:torch.randn(*s,dtype=dtype)
        h,d,e,o,b=16,8,8,5,7
        v=rnd(b,h);w=rnd(h,o)/math.sqrt(h);s=torch.randint(2,(e,h)).to(dtype)*2-1;a=rnd(b,e).softmax(-1)
        Q=torch.linalg.qr(rnd(h,h)).Q@torch.diag(torch.linspace(.7,1.3,h,dtype=dtype))
        for act_name,act in [('gelu',F.gelu),('gelu_tanh',lambda x:F.gelu(x,approximate='tanh')),('silu',F.silu),('relu',F.relu)]:
            record('sign_fusion_'+act_name,seed,dtype,sign_explicit(v,w,s,a,act),sign_fused(v,w,s,a,act))
            record('antipodal_'+act_name,seed,dtype,(mirror(v,Q,act)+mirror(v,-Q,act))/2,v/2)
        x=(rnd(b,d)/math.sqrt(d)).requires_grad_();up=(rnd(d,h)/math.sqrt(d)).requires_grad_();down=(rnd(h,o)/math.sqrt(h)).requires_grad_();router=(rnd(d,e)/math.sqrt(d)).requires_grad_()
        pre=x@up;coef=(x@router).softmax(-1)
        y1=sign_explicit(pre,down,s,coef);y2=sign_fused(pre,down,s,coef)
        g1=torch.autograd.grad(y1.square().mean(),(x,up,down,router),retain_graph=True)
        g2=torch.autograd.grad(y2.square().mean(),(x,up,down,router))
        for gn,p,q in zip(('input','up','down','router'),g1,g2): record('gradient_'+gn,seed,dtype,p,q)
        q,k,val=rnd(3,d),rnd(31,d),rnd(31,6);A=torch.linalg.qr(rnd(d,d)).Q;B=torch.linalg.qr(rnd(6,6)).Q
        record('lazy_cache',seed,dtype,attention(q,k@A,val@B),lazy_attention(q,k,val,A,B))
        record('partition_attention',seed,dtype,attention(q,k,val),partition_attention(q,k[:20],val[:20],k[20:],val[20:]))
        record('QK_gauge',seed,dtype,attention(q@A,k@torch.linalg.inv(A).T,val),attention(q,k,val))
        Bs=rnd(5,6,6);weights=rnd(3,5).softmax(-1)
        outputs=torch.stack([attention(q,k,val@Be) for Be in Bs],1)
        expected=(outputs*weights[:,:,None]).sum(1)
        fused=torch.einsum('bd,bde->be',attention(q,k,val),torch.einsum('bk,kde->bde',weights,Bs))
        record('value_only_fusion',seed,dtype,expected,fused)
        A2=torch.linalg.qr(rnd(d,d)).Q
        record('wrong_average_keys',seed,dtype,(attention(q,k@A,val)+attention(q,k@A2,val))/2,attention(q,k@((A+A2)/2),val),'different')
        qs=torch.stack([torch.linalg.qr(rnd(h,h)).Q for _ in range(4)])
        co=rnd(v.shape[0],4).softmax(-1)
        naive=sum(co[:,i:i+1]*(mirror(v,qs[i])@w) for i in range(4))
        record('general_shared_projection_fusion',seed,dtype,naive,mirror_mixture_fused(v,w,qs,co))
        q2,kl,va,au=rnd(3,8),rnd(31,5),rnd(31,6),rnd(5,8)
        direct=attention(q2,kl@au,va)
        correct=((q2@au.T)@kl.T/math.sqrt(8)).softmax(-1)@va
        record('latent_logical_temperature',seed,dtype,direct,correct)
        record('wrong_latent_temperature',seed,dtype,direct,attention(q2@au.T,kl,va),'different')
with (ROOT/'IDENTITY_METRICS.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
summary={}
for dtype in ('torch.float64','torch.float32'):
    subset=[r for r in rows if r['dtype']==dtype and r['criterion']=='equal']
    summary[dtype]={'checks':len(subset),'maximum_absolute_error':max(r['max_abs_error'] for r in subset),'maximum_scaled_error':max(r['scaled_error'] for r in subset),'pass':all(r['pass_check'] for r in subset)}
    summary[dtype]['max_gradient_error']=max(r['max_abs_error'] for r in subset if r['check'].startswith('gradient'))
summary['total_checks']=len(rows);summary['all_pass']=all(r['pass_check'] for r in rows)
summary['negative_control_min_difference']=min(r['max_abs_error'] for r in rows if r['criterion']=='different')
summary['environment']={'python':platform.python_version(),'torch':torch.__version__,'numpy':__import__('numpy').__version__,'platform':platform.platform(),'device':'CPU','threads':torch.get_num_threads(),'cuda_available':torch.cuda.is_available()}
summary['learned_training_runs']=0;summary['runtime_benchmarks']=0
(ROOT/'MEASUREMENT_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
assert summary['all_pass']
