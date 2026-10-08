import math
import numpy as np
import torch

def make_tasks(seed,rho):
 g=torch.Generator().manual_seed(seed+int(rho*10000));L,Q,A,T,R=6,4,3,8,2
 base=(torch.rand(L,Q,A,generator=g)-.5)*.4
 task=torch.randn(T,R,generator=g)*.65;layer=torch.randn(L,R,generator=g)*.7;qubit=torch.randn(Q,R,generator=g)*.7;axis=torch.randn(A,R,generator=g)*.7
 structured=torch.einsum('tr,lr,qr,ar->tlqa',task,layer,qubit,axis)
 private=torch.randn(T,L,Q,A,generator=g)*.08
 return base.unsqueeze(0)+rho*(structured+.05*private),base

def fit_cp(x,rank,seed,steps=1200):
 torch.manual_seed(seed);x=x.double();T,L,Q,A=x.shape
 factors=[torch.nn.Parameter(torch.randn(d,rank,dtype=torch.float64)*.3) for d in (T,L,Q,A)]
 opt=torch.optim.Adam(factors,lr=.035)
 for _ in range(steps):
  opt.zero_grad();pred=torch.einsum('tr,lr,qr,ar->tlqa',*factors);loss=(pred-x).square().mean();loss.backward();opt.step()
 with torch.no_grad():return [f.detach().clone() for f in factors],float(loss)

def cp_basis(factors):
 _,lf,qf,af=factors
 return torch.stack([lf[:,r,None,None]*qf[None,:,r,None]*af[None,None,:,r] for r in range(lf.shape[1])])

def fit_tt(x,rank):
 # TT-SVD of the training task-angle tensor; returns task core and three gate-mode cores.
 a=x.double();dims=list(a.shape);cores=[];rprev=1
 for mode,d in enumerate(dims[:-1]):
  mat=a.reshape(rprev*d,-1) if mode else a.reshape(d,-1)
  u,s,vh=torch.linalg.svd(mat,full_matrices=False);r=min(rank,len(s))
  core=u[:,:r].reshape(rprev,d,r) if mode else u[:,:r].reshape(d,r)
  cores.append(core);a=(s[:r,None]*vh[:r]);rprev=r
 cores.append(a.reshape(rprev,dims[-1]))
 return cores

def tt_reconstruct(code,cores):
 x=code
 for core in cores:x=torch.tensordot(x,core,dims=([-1],[0]))
 return x

def tt_basis(cores):
 r=cores[0].shape[-1];eye=torch.eye(r,dtype=cores[1].dtype)
 return torch.stack([tt_reconstruct(eye[i],cores[1:]) for i in range(r)])

def fit_linear_basis(train,rank):
 mean=train.mean(0);x=(train-mean).reshape(len(train),-1).double();_,s,vh=torch.linalg.svd(x,full_matrices=False)
 effective=min(rank,len(vh));basis=vh[:effective].reshape(effective,*train.shape[1:]);codes=x@vh[:effective].T
 return mean,basis,codes

def fit_code(target,mean,basis):
 b=basis.reshape(len(basis),-1).T.double();y=(target-mean).reshape(-1).double()
 return torch.linalg.lstsq(b,y).solution.to(target.dtype)

def rotation(axis,theta):
 c=math.cos(theta/2);s=math.sin(theta/2)
 if axis==0:return np.array([[c,-1j*s],[-1j*s,c]],dtype=np.complex128)
 if axis==1:return np.array([[c,-s],[s,c]],dtype=np.complex128)
 return np.array([[np.exp(-1j*theta/2),0],[0,np.exp(1j*theta/2)]],dtype=np.complex128)

def apply_one(v,u,q,n=4):
 shape=(2,)*n;z=np.moveaxis(v.reshape(shape),q,0).reshape(2,-1);out=u@z
 return np.moveaxis(out.reshape(shape),0,q).reshape(-1)

def apply_cnot(v,control,target,n=4):
 out=np.empty_like(v)
 for i,value in enumerate(v):
  if (i>>(n-1-control))&1:j=i^(1<<(n-1-target))
  else:j=i
  out[j]=value
 return out

def unitary(theta):
 arr=np.asarray(theta,dtype=np.float64).reshape(6,4,3);n=4;dim=16;u=np.zeros((dim,dim),dtype=np.complex128)
 for col in range(dim):
  v=np.zeros(dim,dtype=np.complex128);v[col]=1
  for layer in range(6):
   for axis in range(3):
    for q in range(4):v=apply_one(v,rotation(axis,float(arr[layer,q,axis])),q,n)
   for q in range(n-1):v=apply_cnot(v,q,q+1,n)
  u[:,col]=v
 return u

def unitary_metrics(target,pred):
 d=target.shape[0];process=float(abs(np.trace(target.conj().T@pred)/d)**2)
 cols=np.diag(target.conj().T@pred);basis=float(np.mean(np.abs(cols)**2))
 return process,basis
