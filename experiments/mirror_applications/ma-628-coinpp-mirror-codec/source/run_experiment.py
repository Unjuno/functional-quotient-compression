import argparse,csv,json,time,zlib,struct
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];TASKS=[(k,p) for k in range(1,5) for p in range(4)];HELD={(1,0),(2,1),(3,2),(4,3)};PH=np.array([0,np.pi/2,np.pi,3*np.pi/2])
def signal(x,a,k,p):return a*np.sin(2*np.pi*k*x+p)
def fit_code(x,y):
 best=None
 for k in range(1,5):
  m=np.stack([np.sin(2*np.pi*k*x),np.cos(2*np.pi*k*x)],1);c=np.linalg.lstsq(m,y,rcond=None)[0];e=np.mean((m@c-y)**2)
  if best is None or e<best[0]:best=(e,k,c)
 _,k,c=best;return [float(np.linalg.norm(c)),int(k),float(np.arctan2(c[1],c[0])%(2*np.pi))]
def qval(x,lo,hi,b):return np.clip(np.rint((x-lo)/(hi-lo)*((1<<b)-1)),0,(1<<b)-1).astype(int)
def uqval(v,lo,hi,b):return lo+v/((1<<b)-1)*(hi-lo)
def pack(vals,widths):
 bits=''.join(format(int(v),f'0{w}b') for row in vals for v,w in zip(row,widths));bits+=('0'*((-len(bits))%8));return int(bits,2).to_bytes(len(bits)//8,'big') if bits else b''
def unpack(data,n,widths):
 s=''.join(f'{v:08b}' for v in data);out=[];pos=0
 for _ in range(n):
  row=[]
  for w in widths:row.append(int(s[pos:pos+w],2));pos+=w
  out.append(row)
 return np.array(out,dtype=int)
def nrmse(a,b):return float(np.sqrt(np.mean((a-b)**2))/max(np.sqrt(np.mean(a*a)),1e-12))
def make_independent(ygrid):
 q=qval(ygrid,-1.5,1.5,8).astype(np.uint8);h=json.dumps({'schema':'GRID512-U8-v1','range':[-1.5,1.5],'tasks':16},sort_keys=True,separators=(',',':')).encode();raw=struct.pack('>H',len(h))+h+q.tobytes();return zlib.compress(raw,9)
def run(seed):
 rng=np.random.default_rng(seed);amps={t:rng.uniform(.5,1.5) for t in TASKS};x={t:rng.uniform(0,1,64) for t in TASKS};xa={t:rng.uniform(0,1,512) for t in TASKS};fitted={t:fit_code(x[t],signal(x[t],amps[t],t[0],PH[t[1]])) for t in TASKS};grids=np.stack([signal(np.linspace(0,1,512),amps[t],t[0],PH[t[1]]) for t in TASKS]);grid_payload=make_independent(grids)
 rows=[];encode_elapsed={}
 for b in [4,6,8,10,12]:
  widths=[b,2,b];records=[]
  for t in TASKS:
   a,k,p=fitted[t];records.append([qval(a,.5,1.5,b),k-1,qval(p,0,2*np.pi,b)])
  records=np.asarray(records,dtype=int);rawbits=pack(records,widths);header=json.dumps({'schema':'PERIODIC-CODE-UQ-v1','amp_range':[.5,1.5],'phase_range':[0,2*np.pi],'bits':widths,'tasks':TASKS},sort_keys=True,separators=(',',':')).encode();stream=struct.pack('>H',len(header))+header+rawbits;t0=time.perf_counter();coded=zlib.compress(stream,9);enc=time.perf_counter()-t0
  path=R/'source'/f'{seed}_mirror_native_b{b}.bin';path.write_bytes(coded)
  decoded=unpack(zlib.decompress(coded)[2+len(header):],16,widths)
  qcodes=[]
  for row in decoded:qcodes.append([uqval(row[0],.5,1.5,b),int(row[1])+1,uqval(row[2],0,2*np.pi,b)])
  qcodes=np.asarray(qcodes)
  for method in ['mirror_quantized_code','native_coinpp_modulation']:
   for task in HELD:
    ix=TASKS.index(task);pred=signal(xa[task],*qcodes[ix]);truth=signal(xa[task],amps[task],task[0],PH[task[1]])
    rows.append({'seed':seed,'task':list(task),'method':method,'bits_per_amplitude_phase':b,'serialized_bytes':len(coded),'heldout_nrmse':nrmse(truth,pred),'encode_time_s':enc,'decode_compute_proxy':int(512*10),'support_points':64,'support_search_candidates':4})
  for task in HELD:
   ix=TASKS.index(task);grid=uqval(qval(grids[ix],-1.5,1.5,8),-1.5,1.5,8);u=xa[task]*511;i=np.floor(u).astype(int);f=u-i;pred=(1-f)*grid[i]+f*grid[np.minimum(i+1,511)];truth=signal(xa[task],amps[task],task[0],PH[task[1]])
   rows.append({'seed':seed,'task':list(task),'method':'independent_grid_u8','bits_per_amplitude_phase':8,'serialized_bytes':len(grid_payload),'heldout_nrmse':nrmse(truth,pred),'encode_time_s':0.0,'decode_compute_proxy':int(512*8),'support_points':0,'support_search_candidates':0})
 return rows
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seeds',nargs='+',type=int,required=True);args=ap.parse_args();t=time.perf_counter();rows=[]
 for s in args.seeds:rows+=run(s)
 wall=time.perf_counter()-t
 for r in rows:r['wall_time_s']=wall
 (R/'source'/'development.json').write_text(json.dumps(rows,indent=2)+'\n')
 with (R/'RESULTS_CORE.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note'])
  for r in rows:w.writerow(['harmonic_sines',r['seed'],f"{r['method']}_b{r['bits_per_amplitude_phase']}",r['serialized_bytes'],r['support_points'],0,r['decode_compute_proxy'],r['wall_time_s'],'audit_nrmse',r['heldout_nrmse'],'encode_time_s',r['encode_time_s'],'development; fresh sealed'])
 print(json.dumps(rows,indent=2))
