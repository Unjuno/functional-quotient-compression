"""Rank-2 role/query policy factorization on a synthetic Mesh utility world."""
import argparse,json,struct,time
import numpy as np,torch
R=Q=8;K=2;RANK=2;UPDATES=600;LR=.03;MSG=struct.Struct('!BBf')
def payload_factor(A,B):return struct.pack('!4sBBBB32f',b'MA87',1,R,Q,RANK,*A.astype(np.float32).reshape(-1),*B.astype(np.float32).reshape(-1))
def payload_flat(U,mask):return struct.pack('!4sBBB8s64f',b'MA87',1,R,Q,np.packbits(mask.reshape(-1)).tobytes(),*U.astype(np.float32).reshape(-1))
def make_world(seed):
 rng=np.random.default_rng(seed);a=rng.normal(0,.5,(R,RANK));b=rng.normal(0,.5,(Q,RANK));u=a@b.T;mask=rng.random((R,Q))>=.25
 # Ensure each query retains training pairs and has a held-out pair.
 for q in range(Q):
  if mask[:,q].sum()<4:mask[rng.choice(R,4,replace=False),q]=True
  if mask[:,q].all():mask[rng.integers(R),q]=False
 return u,mask
def fit(u,mask,seed):
 torch.manual_seed(seed);A=torch.nn.Parameter(torch.randn(R,RANK)*.1);B=torch.nn.Parameter(torch.randn(Q,RANK)*.1);opt=torch.optim.Adam([A,B],lr=LR);target=torch.tensor(u,dtype=torch.float32);m=torch.tensor(mask)
 for _ in range(UPDATES):pred=A@B.T;loss=((pred-target)[m]**2).mean();opt.zero_grad();loss.backward();opt.step()
 return A.detach().numpy(),B.detach().numpy()
def score(utility,selected):return float(np.mean([1/(1+sum(np.exp(utility[r,q]) for r in selected[q])) for q in range(Q)]))
def run(seed):
 st=time.perf_counter();u,mask=make_world(seed);a,b=fit(u,mask,seed+1000);pred=a@b.T;flat=np.zeros_like(u)
 for q in range(Q):
  vals=u[:,q][mask[:,q]];flat[:,q]=float(vals.mean());flat[mask[:,q],q]=u[mask[:,q],q]
 rng=np.random.default_rng(seed+2000);factor_sel=np.argsort(-pred,axis=0)[:K,:].T;flat_sel=np.argsort(-flat,axis=0)[:K,:].T;oracle_sel=np.argsort(-u,axis=0)[:K,:].T;random_sel=np.stack([rng.choice(R,K,replace=False) for _ in range(Q)])
 held=~mask;mae=float(np.abs(pred[held]-u[held]).mean());factor_n=score(u,factor_sel);flat_n=score(u,flat_sel);oracle_n=score(u,oracle_sel);random_n=score(u,random_sel)
 fp=payload_factor(a,b);tp=payload_flat(flat,mask);op=payload_flat(u,np.ones_like(mask,dtype=bool));random_bytes=struct.pack('!4sBB16B',b'RAND',1,K,*random_sel.astype(np.uint8).reshape(-1));mirror_sel=factor_sel.copy();mirror_diff=int(np.abs(factor_sel-mirror_sel).max());messages=Q*K;central=Q*R
 return {'seed':seed,'heldout_pairs':int(held.sum()),'train_pairs':int(mask.sum()),'heldout_utility_mae':mae,'factorized_nMSE':factor_n,'flat_nMSE':flat_n,'oracle_nMSE':oracle_n,'random_nMSE':random_n,'factorized_payload_bytes':len(fp),'flat_table_payload_bytes':len(tp),'oracle_table_payload_bytes':len(op),'random_policy_payload_bytes':len(random_bytes),'factorized_selected_roles':factor_sel.tolist(),'flat_selected_roles':flat_sel.tolist(),'oracle_selected_roles':oracle_sel.tolist(),'random_selected_roles':random_sel.tolist(),'messages_per_query':K,'total_messages':messages,'central_total_messages':central,'message_bytes':MSG.size,'total_message_bytes':messages*MSG.size,'central_total_message_bytes':central*MSG.size,'policy_scoring_macs':R*Q*RANK,'factor_fit_updates':UPDATES,'latency_hops':1,'native_mirror_role_difference':mirror_diff,'factor_payload_header_bytes':8,'wall_time_s':time.perf_counter()-st}
def main():
 p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');p.add_argument('--fresh',action='store_true');x=p.parse_args();seeds=[97] if x.development else ([601,617,631] if x.fresh else [97,601,617,631]);print(json.dumps({'fixed':{'rank':RANK,'updates':UPDATES,'lr':LR,'heldout_fraction':.25},'runs':[run(s) for s in seeds]},indent=2))
if __name__=='__main__':main()
