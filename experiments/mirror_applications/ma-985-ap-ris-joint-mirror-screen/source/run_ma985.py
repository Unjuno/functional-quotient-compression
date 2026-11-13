"""Small synthetic MU-MISO WMMSE + RIS phase-coordinate screen."""
import argparse,csv,hashlib,io,json,time
from pathlib import Path
import numpy as np
import torch

R=Path(__file__).resolve().parents[1]
M=4;N=8;K=2;CASES=8;PWR=1.;NOISE=.1;DEV=98501;FRESH=(98511,98513,98517);RIS_STEPS=20;OUTER=3;WMMSE_MAX=30
torch.set_num_threads(1)
def bases(ap_rank,ris_rank):
 U=np.stack([np.exp(2j*np.pi*np.arange(M)*j/M)/np.sqrt(M) for j in range(ap_rank)],axis=1).astype(np.complex64)
 x=np.linspace(-1,1,N);B=np.stack([x**j for j in range(ris_rank)],axis=1).astype(np.float32)
 return U,B

def steer(n,u):return np.exp(1j*np.pi*np.arange(n)*u)/np.sqrt(n)

def make_world(seed):
 rng=np.random.default_rng(seed);rows=[]
 for _ in range(CASES):
  G=.8*np.outer(steer(N,rng.uniform(-.8,.8)),steer(M,rng.uniform(-.8,.8)).conj())+.2*(rng.normal(size=(N,M))+1j*rng.normal(size=(N,M)))/np.sqrt(2*M)
  d=[];r=[]
  for k in range(K):
   u=rng.uniform(-.9,.9);v=rng.uniform(-.9,.9)
   d.append(.25*steer(M,u)+.08*(rng.normal(size=M)+1j*rng.normal(size=M))/np.sqrt(2*M))
   r.append(.7*steer(N,v)+.1*(rng.normal(size=N)+1j*rng.normal(size=N))/np.sqrt(2*N))
  rows.append({'G':G.astype(np.complex64),'d':np.asarray(d,dtype=np.complex64),'r':np.asarray(r,dtype=np.complex64)})
 return rows

def eff(ch,theta):
 phase=np.exp(1j*theta);return ch['d']+(ch['r'].conj()*phase[None,:])@ch['G']

def wsr(H,W):
 t=H@W;sig=np.abs(np.diag(t))**2;tot=np.sum(np.abs(t)**2,axis=1)+NOISE
 return float(np.sum(np.log2(1+sig/np.maximum(tot-sig,1e-12))))

def solve_wmmse(H,max_iter=WMMSE_MAX):
 p=H.conj().T.copy();p*=np.sqrt(PWR/max(np.sum(np.abs(p)**2),1e-12));old=-1.;V=p;used=0
 for it in range(max_iter):
  T=H@V;tot=np.sum(np.abs(T)**2,axis=1)+NOISE;sig=np.diag(T);u=sig/tot
  mse=np.maximum(1-2*np.real(np.conj(u)*sig)+np.abs(u)**2*tot,1e-9);wt=1/mse
  A=np.zeros((H.shape[1],H.shape[1]),np.complex128);rhs=np.zeros((H.shape[1],K),np.complex128)
  for k in range(K):
   hk=H[k];A+=wt[k]*np.abs(u[k])**2*np.outer(hk.conj(),hk);rhs[:,k]=wt[k]*u[k]*hk.conj()
  I=np.eye(H.shape[1],dtype=np.complex128)
  def make(mu):return np.linalg.solve(A+mu*I,rhs)
  V=make(0.);power=np.sum(np.abs(V)**2)
  if power>PWR:
   lo,hi=0.,1.
   while np.sum(np.abs(make(hi))**2)>PWR:hi*=2
   for _ in range(35):
    mid=(lo+hi)/2
    if np.sum(np.abs(make(mid))**2)>PWR:lo=mid
    else:hi=mid
   V=make(hi)
  rate=wsr(H,V);used=it+1
  if abs(rate-old)<1e-6:break
  old=rate
 return V.astype(np.complex64),used,float(max(np.sum(np.abs(V)**2)-PWR,0.))

def opt_phase(ch,W,basis,init=None):
 dim=N if basis is None else basis.shape[1]
 q=torch.nn.Parameter(torch.zeros(dim,dtype=torch.float32) if init is None else torch.tensor(init,dtype=torch.float32))
 opt=torch.optim.Adam([q],lr=.06);G=torch.tensor(ch['G']);d=torch.tensor(ch['d']);r=torch.tensor(ch['r']);W=torch.tensor(W)
 B=None if basis is None else torch.tensor(basis)
 best=q.detach().clone();best_loss=float('inf')
 for _ in range(RIS_STEPS):
  opt.zero_grad(set_to_none=True);theta=q if B is None else B@q
  hh=d+(r.conj()*torch.exp(1j*theta)[None,:])@G;T=hh@W;sig=torch.abs(torch.diagonal(T))**2;tot=torch.sum(torch.abs(T)**2,dim=1)+NOISE
  loss=-torch.sum(torch.log2(1+sig/(tot-sig+1e-9)))
  if float(loss.detach())<best_loss:best_loss=float(loss.detach());best=q.detach().clone()
  loss.backward();opt.step()
 return best.numpy()

def optimize(ch,U,basis):
 q=np.zeros(N if basis is None else basis.shape[1],np.float32);theta=q if basis is None else basis@q;iters=0;resid=0.
 for _ in range(OUTER):
  H=eff(ch,theta)@U;V,it,power=solve_wmmse(H);W=U@V;iters+=it;resid=max(resid,power)
  q=opt_phase(ch,W,basis,q);theta=q if basis is None else basis@q
 H=eff(ch,theta)@U;V,it,power=solve_wmmse(H);W=U@V;iters+=it;resid=max(resid,power)
 return W.astype(np.complex64),theta.astype(np.float32),q.astype(np.float32),iters,resid

def serial_payload(method,Ws,Thetas,Codes,U,B):
 meta={'method':'factorized-control' if method in ('mirror','native_product') else method,'ap_antennas':M,'ris_elements':N,'users':K,'config_ids':list(range(CASES))}
 if method=='full':state={'metadata':meta,'W':torch.tensor(Ws),'theta':torch.tensor(Thetas)}
 elif method=='ap_low_fullris':state={'metadata':meta,'ap_basis':torch.tensor(U),'ap_code':torch.tensor(np.stack([U.conj().T@w for w in Ws])),'theta':torch.tensor(Thetas)}
 elif method=='fullap_ris_low':state={'metadata':meta,'W':torch.tensor(Ws),'ris_basis':torch.tensor(B),'ris_code':torch.tensor(Codes)}
 else:state={'metadata':meta,'ap_basis':torch.tensor(U),'ris_basis':torch.tensor(B),'ap_code':torch.tensor(np.stack([U.conj().T@w for w in Ws])),'ris_code':torch.tensor(Codes)}
 b=io.BytesIO();torch.save(state,b);return b.getvalue()

def score_decoded(method,payload,channels):
 st=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False);rates=[];powers=[]
 for i,ch in enumerate(channels):
  if method=='full':W=st['W'][i].numpy();theta=st['theta'][i].numpy()
  elif method=='ap_low_fullris':W=st['ap_basis'].numpy()@st['ap_code'][i].numpy();theta=st['theta'][i].numpy()
  elif method=='fullap_ris_low':W=st['W'][i].numpy();theta=st['ris_basis'].numpy()@st['ris_code'][i].numpy()
  else:W=st['ap_basis'].numpy()@st['ap_code'][i].numpy();theta=st['ris_basis'].numpy()@st['ris_code'][i].numpy()
  H=eff(ch,theta);rates.append(wsr(H,W));powers.append(float(max(np.sum(np.abs(W)**2)-PWR,0.)))
 return float(np.mean(rates)),float(max(powers))

def evaluate(seed,ap_rank=2,ris_rank=2):
 Ured,Bred=bases(ap_rank,ris_rank);channels=make_world(seed);methods={'full':(np.eye(M,dtype=np.complex64),None),'ap_low_fullris':(Ured,None),'fullap_ris_low':(np.eye(M,dtype=np.complex64),Bred),'mirror':(Ured,Bred),'native_product':(Ured,Bred)};out=[]
 for method,(U,basis) in methods.items():
  start=time.perf_counter();Ws=[];Thetas=[];Codes=[];iters=0;maxp=0.
  for ch in channels:
   W,theta,q,it,pr=optimize(ch,U,basis);Ws.append(W);Thetas.append(theta);Codes.append(q);iters+=it;maxp=max(maxp,pr)
  payload=serial_payload(method,Ws,Thetas,Codes,Ured,Bred);rate,replay_power=score_decoded(method,payload,channels)
  red=U.shape[1];macs=iters*(red**3+K*red**2)+CASES*OUTER*RIS_STEPS*N*M*K
  out.append({'condition':method,'mean_wsr':rate,'max_power_residual':max(maxp,replay_power),'payload_bytes':len(payload),'ap_complex_coefficients':int(red*K),'ris_real_coordinates':int(N if basis is None else basis.shape[1]),'ap_basis_rank':int(red),'ris_basis_rank':0 if basis is None else int(basis.shape[1]),'wmmse_iterations':iters,'ris_gradient_steps':OUTER*RIS_STEPS*CASES,'complex_mac_proxy':macs,'wall_s':time.perf_counter()-start,'payload_sha256':hashlib.sha256(payload).hexdigest()})
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev-only',action='store_true');ap.add_argument('--ap-rank',type=int,default=2);ap.add_argument('--ris-rank',type=int,default=2);args=ap.parse_args()
 dev=evaluate(DEV,args.ap_rank,args.ris_rank);(R/'source'/'DEVELOPMENT.json').write_text(json.dumps({'seed':DEV,'ap_rank':args.ap_rank,'ris_rank':args.ris_rank,'results':dev},indent=2)+'\n')
 if args.dev_only:print(json.dumps(dev));return
 fresh=[];rows=[]
 for seed in FRESH:
  z=evaluate(seed,args.ap_rank,args.ris_rank);fresh.append({'seed':seed,'results':z})
  for x in z:rows.append({'seed':seed,'condition':x['condition'],'mean_weighted_sum_rate':x['mean_wsr'],'max_power_residual':x['max_power_residual'],'payload_bytes':x['payload_bytes'],'ap_complex_coefficients':x['ap_complex_coefficients'],'ris_real_coordinates':x['ris_real_coordinates'],'ap_basis_rank':x['ap_basis_rank'],'ris_basis_rank':x['ris_basis_rank'],'wmmse_iterations':x['wmmse_iterations'],'ris_gradient_steps':x['ris_gradient_steps'],'complex_mac_proxy_total':x['complex_mac_proxy'],'wall_time_s':x['wall_s']})
 with (R/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n').writerows(rows)
 (R/'source'/'FRESH.json').write_text(json.dumps(fresh,indent=2)+'\n');print(json.dumps(fresh))
if __name__=='__main__':main()
