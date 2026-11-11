"""Observation-only Mesh admission policy on a linear-Gaussian world."""
import argparse,json,struct,time
import numpy as np
SIGMAS=np.array([.25,.5,.75,1.0],dtype=np.float32);DIMS=8;REPS=4;AGENTS=DIMS*REPS;N=4096
MESSAGE=struct.Struct('!BBf')
def schema_payload():return struct.pack('!4sBBBB4f',b'MESH',1,DIMS,REPS,len(SIGMAS),*map(float,SIGMAS))
def mirror_policy_payload():
 ids=bytes([REPS*j for j in range(DIMS)])
 return struct.pack('!4sBB',b'MA86',1,DIMS)+ids
def native_rule_payload():return struct.pack('!4sBBBB4f',b'RULE',1,DIMS,REPS,len(SIGMAS),*map(float,SIGMAS))
def best_agent(j):return REPS*j
def posterior_single(y,sigma):
 p=1/(sigma*sigma);return (p/(1+p))*y
def run(seed):
 t=time.perf_counter();rng=np.random.default_rng(seed);z=rng.normal(size=(N,DIMS));noise=rng.normal(size=(N,DIMS,REPS))*SIGMAS;obs=z[:,:,None]+noise
 best=[];random_ids=[];records=[]
 for j in range(DIMS):
  i=best_agent(j);best.append(i);ri=REPS*j+int(rng.integers(REPS));random_ids.append(ri)
  central_precision=np.sum(1/SIGMAS**2);central=(np.sum(obs[:,j,:]/SIGMAS[None,:]**2,axis=1)/(1+central_precision));single=posterior_single(obs[:,j,i%REPS],SIGMAS[i%REPS]);rr=posterior_single(obs[:,j,ri%REPS],SIGMAS[ri%REPS]);
  records.append({'query':j,'central_normalized_mse':float(np.mean((central-z[:,j])**2)),'best_single_normalized_mse':float(np.mean((single-z[:,j])**2)),'random_single_normalized_mse':float(np.mean((rr-z[:,j])**2)),'best_agent_id':i,'random_agent_id':ri,'central_messages':AGENTS,'selected_messages':1,'central_message_bytes':AGENTS*MESSAGE.size,'selected_message_bytes':MESSAGE.size,'central_scalar_updates':AGENTS*3,'selected_scalar_updates':3,'hop_count_central':1,'hop_count_selected':1})
 schema=schema_payload();policy=mirror_policy_payload();native=native_rule_payload();
 # Explicitly encode a representative selected observation to verify actual wire-format size.
 example=MESSAGE.pack(best[0],0,float(obs[0,0,0]));
 return {'seed':seed,'latent_draws':N,'per_query':records,'mean_central_nMSE':float(np.mean([x['central_normalized_mse'] for x in records])),'mean_best_single_nMSE':float(np.mean([x['best_single_normalized_mse'] for x in records])),'mean_random_single_nMSE':float(np.mean([x['random_single_normalized_mse'] for x in records])),'total_central_messages':AGENTS*DIMS,'total_selected_messages':DIMS,'total_central_message_bytes':AGENTS*DIMS*MESSAGE.size,'total_selected_message_bytes':DIMS*MESSAGE.size,'schema_bytes':len(schema),'mirror_policy_bytes':len(policy),'native_rule_bytes':len(native),'combined_mirror_program_bytes':len(schema)+len(policy),'message_bytes_each':MESSAGE.size,'message_example_hex':example.hex(),'max_mirror_native_agent_id_difference':max(abs(int(mirror_policy_payload()[6+i])-best_agent(i)) for i in range(DIMS)),'latency_hops':1,'total_scalar_updates_central':AGENTS*DIMS*3,'total_scalar_updates_selected':DIMS*3,'wall_time_s':time.perf_counter()-t}
def main():
 p=argparse.ArgumentParser();p.add_argument('--development',action='store_true');p.add_argument('--fresh',action='store_true');a=p.parse_args();seeds=[89] if a.development else ([503,521,547] if a.fresh else [89,503,521,547]);print(json.dumps({'protocol':{'dims':DIMS,'agents':AGENTS,'sigmas':SIGMAS.tolist(),'message_bytes':MESSAGE.size},'runs':[run(s) for s in seeds]},indent=2))
if __name__=='__main__':main()
