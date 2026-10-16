"""GIC001: exact small spline-bank learning, native controls and PCGrad.

All science arithmetic is NumPy float64. Targets and gradient statistics are
normalized using training observations only. No remote data or accelerators.
"""
from __future__ import annotations
import json
import struct
from pathlib import Path
from typing import Any
import numpy as np

TASKS = 8
FAMILIES = ('identical', 'aligned', 'partial', 'unrelated')


def basis(x: np.ndarray, knots: int = 17) -> np.ndarray:
    """Uniform linear-spline hat basis, only on the closed [-1,1] domain."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if knots < 2 or not np.all(np.isfinite(x)) or np.any(np.abs(x) > 1 + 1e-14):
        raise ValueError('finite inputs in [-1,1] and >=2 knots are required')
    grid = np.linspace(-1, 1, knots)
    return np.maximum(1 - np.abs(x[:, None] - grid[None, :]) * (knots - 1) / 2, 0)


def make_world(family: str, seed: int, ntrain: int = 128,
               nval: int = 128, ntest: int = 512) -> dict[str, Any]:
    """Generate disjoint input draws and known-rank nonlinear curve families."""
    if family not in FAMILIES:
        raise ValueError(family)
    rng = np.random.default_rng(np.random.SeedSequence([seed, FAMILIES.index(family)]))
    k = 17
    def smooth(shape: tuple[int, ...]) -> np.ndarray:
        a = rng.normal(size=shape)
        return (2*a + np.roll(a,1,axis=0) + np.roll(a,-1,axis=0))/4
    c0 = smooth((k,))
    d, _ = np.linalg.qr(smooth((k, 2)))
    d *= np.sqrt(k)
    codes = rng.normal(scale=.65, size=(TASKS,2)); codes -= codes.mean(0)
    truth = []
    if family == 'identical':
        c = np.repeat(c0[None,:], TASKS, axis=0)
    elif family == 'unrelated':
        c = smooth((k,TASKS)).T
    else:
        c = c0[None,:] + codes @ d.T
        if family == 'partial':
            truth = sorted(rng.choice(TASKS,2,replace=False).tolist())
            for t in truth:
                r = smooth((k,)); r -= d @ (d.T @ r) / k
                r *= .8*np.sqrt(k)/np.linalg.norm(r)
                c[t] += r
    w: dict[str,Any] = {'family':family,'world':seed,'private_truth':truth}
    for name,n in [('train',ntrain),('val',nval),('test',ntest)]:
        # Stratification keeps the train basis well conditioned without reusing inputs.
        x = -1 + 2*(np.arange(n)+rng.random(n))/n
        rng.shuffle(x)
        w['x'+name] = x; w['B'+name] = basis(x,k)
        w['Y'+name] = w['B'+name] @ c.T
    scale = float(np.sqrt(np.mean(w['Ytrain']**2)))
    for name in ('train','val','test'):
        w['Y'+name] /= scale
    w['teacher_coefficients'] = c/scale
    w['scale'] = scale
    w['H'] = w['Btrain'].T @ w['Btrain']/ntrain
    w['Q'] = w['Ytrain'].T @ w['Btrain']/ntrain
    return w


def kind(name: str) -> str:
    if name.startswith('shared'): return 'shared'
    if name == 'independent': return 'independent'
    if name == 'film': return 'film'
    return 'lowrank'


def initialize(name: str, tasks: int, knots: int, seed: int,
               private_ids: list[int] | None = None) -> dict[str,np.ndarray]:
    if tasks != TASKS:
        raise ValueError('this frozen fixture has exactly 8 tasks')
    rng = np.random.default_rng(seed)
    c0 = rng.normal(scale=.01,size=knots)
    mode = kind(name)
    if mode == 'independent': return {'C':np.repeat(c0[None,:],tasks,axis=0)}
    p = {'c0':c0}
    if mode == 'film':
        p.update(a=np.ones(tasks), b=np.zeros(tasks))
    elif mode == 'lowrank':
        r = 3 if name == 'rank3' else 2
        p.update(D=rng.normal(scale=.1,size=(knots,r)), M=np.zeros((tasks,r)))
        if 'private' in name:
            ids = np.array([0,1] if private_ids is None else private_ids, dtype=np.int64)
            if len(ids)!=2 or len(np.unique(ids))!=2 or np.any((ids<0)|(ids>=tasks)):
                raise ValueError('two distinct valid private IDs required')
            p.update(P=np.zeros((2,knots)),ids=ids)
    return p


def coefficients(name: str, p: dict[str,np.ndarray]) -> np.ndarray:
    mode = kind(name)
    if mode == 'independent': return p['C'].copy()
    if mode == 'film': return p['a'][:,None]*p['c0'][None,:]+p['b'][:,None]
    c = np.repeat(p['c0'][None,:],TASKS,axis=0)
    if mode == 'lowrank': c += p['M'] @ p['D'].T
    if 'P' in p: c[p['ids']] += p['P']
    return c


def shared_keys(name: str) -> list[str]:
    return {'shared':['c0'],'film':['c0'],'lowrank':['c0','D'],'independent':[]}[kind(name)]


def gradients(name: str, p: dict[str,np.ndarray], hessian: np.ndarray,
              q: np.ndarray) -> tuple[dict[str,np.ndarray],np.ndarray,np.ndarray]:
    """Exact derivatives of per-task full-batch MSE, NOT a numerical estimate.

    Returns mean-loss parameter derivatives, T x shared_dimension gradients,
    and T x knot_count coefficient derivatives of EACH task (not /T).
    """
    h = 2*(coefficients(name,p) @ hessian-q)
    mode = kind(name)
    if mode == 'independent': return {'C':h/TASKS}, np.empty((TASKS,0)), h
    if mode == 'film':
        gs = h*p['a'][:,None]
        g = {'c0':gs.mean(0),'a':(h*p['c0'][None,:]).sum(1)/TASKS,'b':h.sum(1)/TASKS}
        return g,gs,h
    g = {'c0':h.mean(0)}; gs = h
    if mode == 'lowrank':
        gd = h[:,:,None]*p['M'][:,None,:]
        g.update(D=gd.mean(0),M=h@p['D']/TASKS)
        gs = np.concatenate([h,gd.reshape(TASKS,-1)],axis=1)
        if 'P' in p: g['P'] = h[p['ids']]/TASKS
    return g,gs,h


def pcgrad(g: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """PCGrad on shared blocks only; compare with ORIGINAL other gradients."""
    out = g.copy()
    norm2 = np.sum(g*g,axis=1)
    for i in range(len(g)):
        for j in rng.permutation(len(g)):
            dot = float(out[i] @ g[j])
            if dot < 0 and norm2[j] > 1e-30:
                out[i] -= dot / norm2[j] * g[j]
    return out.mean(0)


def project_into(g: dict[str,np.ndarray], shared: np.ndarray, name: str,
                 rng: np.random.Generator) -> dict[str,np.ndarray]:
    out = dict(g); flat = pcgrad(shared,rng); offset=0
    for key in shared_keys(name):
        n=g[key].size; out[key]=flat[offset:offset+n].reshape(g[key].shape); offset+=n
    return out


def diagnostics(g: np.ndarray) -> dict[str, Any]:
    norms = np.linalg.norm(g,axis=1)
    denom = float(norms.mean()); valid = norms>1e-10
    if g.shape[1]==0 or denom<=1e-10:
        return {'cancel_ratio':None,'conflict_fraction':None,'mean_grad_norm':denom,'valid_tasks':int(valid.sum())}
    unit = g[valid]/norms[valid,None]
    dots = unit@unit.T; pairs=dots[np.triu_indices(len(unit),1)]
    return {'cancel_ratio':float(np.linalg.norm(g.mean(0))/denom),
            'conflict_fraction':float(np.mean(pairs < -1e-8)) if len(pairs) else None,
            'mean_grad_norm':denom,'valid_tasks':int(valid.sum())}


class Adam:
    """Small transparent Adam, checked independently against torch.optim.Adam."""
    def __init__(self,p:dict[str,np.ndarray],lr:float):
        self.lr=lr; self.step=0
        self.m={k:np.zeros_like(v) for k,v in p.items() if k!='ids'}
        self.v={k:np.zeros_like(v) for k,v in p.items() if k!='ids'}
    def update(self,p:dict[str,np.ndarray],g:dict[str,np.ndarray]) -> None:
        self.step+=1
        for k,grad in g.items():
            self.m[k]=.9*self.m[k]+.1*grad
            self.v[k]=.999*self.v[k]+.001*grad*grad
            p[k] -= self.lr*(self.m[k]/(1-.9**self.step))/(np.sqrt(self.v[k]/(1-.999**self.step))+1e-8)


def oracle(b:np.ndarray,y:np.ndarray,rank:int,
           private_ids:list[int]|np.ndarray|None=None) -> np.ndarray:
    """Global empirical MSE optimum: QR-weighted reduced-rank regression.

    Private tasks are unrestricted rows. The remaining task functions share
    a mean curve plus a centered rank-r correction. Never consult test data.
    """
    u,r=np.linalg.qr(b,mode='reduced')
    if np.linalg.matrix_rank(r)<b.shape[1]: raise ValueError('train basis must have full column rank')
    z=y.T@u
    ids=[] if private_ids is None else list(private_ids)
    keep=np.array([i for i in range(y.shape[1]) if i not in ids])
    zhat=z.copy(); mean=z[keep].mean(0)
    a,s,v=np.linalg.svd(z[keep]-mean,full_matrices=False)
    zhat[keep]=mean+(a[:,:rank]*s[:rank])@v[:rank]
    return np.linalg.solve(r,zhat.T).T


def save_model(path:Path,name:str,p:dict[str,np.ndarray]) -> int:
    """Lossless native codec; names that do not affect inference cost no bytes."""
    arrays=[]; chunks=[]
    for key in sorted(p):
        dtype='<i8' if key=='ids' else '<f8'
        a=np.asarray(p[key],dtype=dtype)
        arrays.append({'key':key,'shape':list(a.shape),'dtype':dtype,'bytes':a.nbytes})
        chunks.append(a.tobytes(order='C'))
    header=json.dumps({'version':1,'kind':kind(name),'tasks':TASKS,'domain':[-1.,1.],
                       'basis':'uniform-linear-hat','arrays':arrays},sort_keys=True,separators=(',',':')).encode()
    payload=b'GIC1'+struct.pack('<I',len(header))+header+b''.join(chunks)
    path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(payload)
    return len(payload)


def load_model(path:Path) -> tuple[str,dict[str,np.ndarray]]:
    data=path.read_bytes()
    if len(data)<8 or data[:4]!=b'GIC1': raise ValueError('bad model header')
    n=struct.unpack('<I',data[4:8])[0]
    if n>100000 or 8+n>len(data): raise ValueError('invalid header length')
    try: h=json.loads(data[8:8+n])
    except Exception as e: raise ValueError('invalid metadata') from e
    if h.get('version')!=1 or h.get('tasks')!=TASKS or h.get('basis')!='uniform-linear-hat':
        raise ValueError('unsupported model schema')
    offset=8+n; p={}
    for a in h['arrays']:
        shape=tuple(a['shape']); size=int(np.prod(shape)); dtype=a['dtype']
        if dtype not in ('<f8','<i8') or size*8!=a['bytes'] or offset+a['bytes']>len(data):
            raise ValueError('bad array schema')
        p[a['key']]=np.frombuffer(data,dtype=dtype,count=size,offset=offset).reshape(shape).copy()
        if not np.isfinite(p[a['key']]).all(): raise ValueError('nonfinite model')
        offset+=a['bytes']
    if offset!=len(data): raise ValueError('trailing or missing model bytes')
    return h['kind'],p
