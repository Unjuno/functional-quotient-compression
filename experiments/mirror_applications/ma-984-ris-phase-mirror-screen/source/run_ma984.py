"""Ideal RIS phase-control serialization and rate simulation."""
import argparse,csv,hashlib,json,struct,time
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parents[1]
N=64;GROUP=8;SCENARIOS=128;SNR=10.;DEV=98401;FRESH=(98411,98413,98417)
METHODS={'continuous_full_element_upper':1,'native_full_element_2bit':2,'native_8_group_2bit':3,'mirror_two_coordinate_5bit':4,'native_two_coordinate_phase_gradient_5bit':5}
HEADER=struct.Struct('<8sBBHI')

def bitpack(v,bits):
 acc=0;nb=0;out=bytearray()
 for x in v:
  acc|=(int(x)&((1<<bits)-1))<<nb;nb+=bits
  while nb>=8:out.append(acc&255);acc>>=8;nb-=8
 if nb:out.append(acc&255)
 return bytes(out)

def qphase(x,bits):
 levels=1<<bits;return np.mod(np.rint((np.mod(x+np.pi,2*np.pi)/(2*np.pi))*levels).astype(int),levels)

def dphase(q,bits):return (np.asarray(q,dtype=float)+.5)*(2*np.pi/(1<<bits))-np.pi

def make_scenarios(seed):
 rng=np.random.default_rng(seed);x=np.linspace(-1,1,N);out=[]
 for _ in range(SCENARIOS):
  slope=rng.uniform(-1.7,1.7);curve=rng.normal(0,.07);offset=rng.uniform(-np.pi,np.pi);noise=rng.normal(0,.04,N)
  phase=offset+slope*x+curve*x*x+noise;amp=np.clip(rng.normal(1,.10,N),.65,1.35)*.7/N
  c=amp*np.exp(1j*phase);hd=.25*np.exp(1j*rng.uniform(-np.pi,np.pi));target=np.angle(hd)-np.angle(c)
  out.append((hd,c,target))
 return out

def fit_linear(target):
 x=np.linspace(-1,1,N);y=np.unwrap(target);return np.linalg.lstsq(np.stack([np.ones(N),x],axis=1),y,rcond=None)[0]

def phases_for(scenarios,method):
 allp=[];codes=[]
 for hd,c,target in scenarios:
  if method=='continuous_full_element_upper':theta=target
  elif method=='native_full_element_2bit':codes.append(qphase(target,2));theta=dphase(codes[-1],2)
  elif method=='native_8_group_2bit':
   q=[]
   for i in range(0,N,GROUP):
    z=np.mean(np.exp(1j*target[i:i+GROUP]));q.append(int(qphase(np.angle(z),2)))
   codes.append(np.asarray(q));theta=np.repeat(dphase(codes[-1],2),GROUP)
  else:
   params=fit_linear(target);q=qphase(params,5);codes.append(q);dec=dphase(q,5);x=np.linspace(-1,1,N);theta=dec[0]+dec[1]*x
  allp.append(theta)
 return np.asarray(allp),codes

def encode(method,codes,phases):
 if method=='continuous_full_element_upper':body=np.asarray(phases,dtype='<f4').tobytes();bits=32*N
 elif method=='native_full_element_2bit':body=bitpack(np.concatenate(codes),2);bits=2*N
 elif method=='native_8_group_2bit':body=bitpack(np.concatenate(codes),2);bits=2*(N//GROUP)
 else:body=bitpack(np.concatenate(codes),5);bits=10
 header=HEADER.pack(b'MA984RIS',1,METHODS[method],SCENARIOS,bits)
 return header+body,bits

def decode_payload(method,payload):
 magic,ver,mid,n,bits=HEADER.unpack(payload[:HEADER.size]);assert magic==b'MA984RIS' and ver==1 and mid==METHODS[method] and n==SCENARIOS
 body=payload[HEADER.size:]
 if method=='continuous_full_element_upper':return np.frombuffer(body,dtype='<f4').reshape(SCENARIOS,N).astype(float)
 width=2 if method in ('native_full_element_2bit','native_8_group_2bit') else 5
 nvalues={'native_full_element_2bit':N,'native_8_group_2bit':N//GROUP,'mirror_two_coordinate_5bit':2,'native_two_coordinate_phase_gradient_5bit':2}[method]
 k=nvalues*SCENARIOS;v=[];acc=0;nb=0
 for byte in body:
  acc|=byte<<nb;nb+=8
  while nb>=width and len(v)<k:v.append(acc&((1<<width)-1));acc>>=width;nb-=width
 arr=np.asarray(v).reshape(SCENARIOS,-1)
 if method=='native_full_element_2bit':return dphase(arr,2)
 if method=='native_8_group_2bit':return np.repeat(dphase(arr,2),GROUP,axis=1)
 coeff=dphase(arr,5);x=np.linspace(-1,1,N);return coeff[:,0,None]+coeff[:,1,None]*x[None,:]

def eval_rate(scenarios,theta):
 vals=[]
 for (hd,c,_),t in zip(scenarios,theta):vals.append(np.log2(1+SNR*np.abs(hd+np.sum(c*np.exp(1j*t)))**2))
 return np.asarray(vals)

def evaluate(seed):
 t0=time.perf_counter();sc=make_scenarios(seed);out=[];details=[]
 for method in METHODS:
  tmethod=time.perf_counter()
  theta,codes=phases_for(sc,method);payload,bits=encode(method,codes,theta);decoded=decode_payload(method,payload);rates=eval_rate(sc,decoded)
  controls={'continuous_full_element_upper':N,'native_full_element_2bit':N,'native_8_group_2bit':N,'mirror_two_coordinate_5bit':N,'native_two_coordinate_phase_gradient_5bit':N}[method]
  independent={'continuous_full_element_upper':N,'native_full_element_2bit':N,'native_8_group_2bit':N//GROUP,'mirror_two_coordinate_5bit':2,'native_two_coordinate_phase_gradient_5bit':2}[method]
  out.append({'condition':method,'mean_rate':float(rates.mean()),'p5_rate':float(np.percentile(rates,5)),'bits':bits,'bytes':len(payload),'physical_updates':controls,'independent_controls':independent,'evaluations':N,'wall_s':time.perf_counter()-tmethod,'payload_sha256':hashlib.sha256(payload).hexdigest(),'rates':rates.tolist()})
  details.append({'condition':method,'payload_bytes':len(payload),'payload_sha256':hashlib.sha256(payload).hexdigest(),'phase_command_bits_per_scenario':bits})
 return out,details

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dev-only',action='store_true');a=ap.parse_args()
 d,_=evaluate(DEV);(R/'source'/'DEVELOPMENT.json').write_text(json.dumps({'seed':DEV,'results':d},indent=2)+'\n')
 if a.dev_only:print(json.dumps(d));return
 fresh=[];rows=[]
 for seed in FRESH:
  z,detail=evaluate(seed);fresh.append({'seed':seed,'results':z,'payloads':detail})
  for x in z:rows.append({'seed':seed,'condition':x['condition'],'mean_spectral_efficiency':x['mean_rate'],'p5_spectral_efficiency':x['p5_rate'],'command_bits_per_scenario':x['bits'],'serialized_payload_bytes':x['bytes'],'physical_elements_updated':x['physical_updates'],'independent_phase_codes':x['independent_controls'],'controller_evaluations':x['evaluations'],'wall_time_s':x['wall_s']})
 with (R/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n').writerows(rows)
 (R/'source'/'FRESH.json').write_text(json.dumps(fresh,indent=2)+'\n');print(json.dumps(fresh))
if __name__=='__main__':main()
