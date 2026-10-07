#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import PacketDecoder,compute_proxy
ROOT=Path(__file__).resolve().parents[1];FIELDS=["condition","world_or_seed","mode","method","serialized_model_bytes","packet_address_bits","combined_cost_bytes","train_tokens_or_examples","optimizer_updates","active_compute_proxy","wall_time_s","primary_metric","primary_value","secondary_metric","secondary_value","status_note"]
P=4;STATES=16;RULES=8;UPDATES=1200

def make_world(world):
    rng=np.random.default_rng(world);return torch.tensor(np.stack([[rng.permutation(STATES) for _ in range(RULES)] for _ in range(2)]),dtype=torch.long)
def sample(table,n,mode,seed):
    g=torch.Generator().manual_seed(seed);rule=torch.randint(RULES,(n,),generator=g);start=torch.randint(STATES,(n,),generator=g)
    if mode=="correlated_two_packet_codes":z=torch.randint(2,(n,1),generator=g).expand(-1,P).clone()
    elif mode=="independent_sixteen_packet_codes":z=torch.randint(2,(n,P),generator=g)
    else:raise ValueError(mode)
    ys=[];cur=start.clone()
    for j in range(P):cur=table[z[:,j],rule,cur];ys.append(cur.clone())
    code=(z*(2**torch.arange(P-1,-1,-1))).sum(-1)
    if mode=="correlated_two_packet_codes":code=z[:,0].clone()
    return rule,start,z,code,torch.stack(ys,1)
def evaluate(model,batch,table):
    rule,start,bits,code,target=batch;model.eval()
    with torch.no_grad():
        logits=model(rule,start,bits,code);nll=float(F.cross_entropy(logits.reshape(-1,STATES),target.reshape(-1)));pred=logits.argmax(-1)
        tok=float((pred==target).float().mean());joint=float((pred==target).all(-1).float().mean());prev=start;valid=torch.ones_like(start,dtype=torch.bool)
        for j in range(P):
            y=pred[:,j];valid&=(table[0,rule,prev]==y)|(table[1,rule,prev]==y);prev=y
    return {"token_nll":nll,"token_accuracy":tok,"joint_packet_accuracy":joint,"valid_path_rate":float(valid.float().mean())}
def fit(method,mode,table,seed,lr,updates=UPDATES):
    codes=2 if mode=="correlated_two_packet_codes" else 16;torch.manual_seed(seed);model=PacketDecoder(method,P,codes,STATES,RULES);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4);g=torch.Generator().manual_seed(seed+29);start_time=time.perf_counter();model.train()
    for _ in range(updates):
        batch=sample(table,64,mode,int(torch.randint(2**31-1,(1,),generator=g)));r,s,b,c,y=batch;logits=model(r,s,b,c);loss=F.cross_entropy(logits.reshape(-1,STATES),y.reshape(-1));opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return model,time.perf_counter()-start_time
def address_bits(mode,method):return 1 if mode=="correlated_two_packet_codes" else P
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);ap.add_argument('--lr',type=float);a=ap.parse_args();torch.set_num_threads(1)
    methods=PacketDecoder.METHODS;modes=['correlated_two_packet_codes','independent_sixteen_packet_codes']
    if a.phase=='dev':worlds=[(24800,248000)];lrs=[.003,.01]
    else:
        if a.lr not in (.003,.01):raise SystemExit('fresh requires frozen --lr')
        worlds=[(24801,248001),(24802,248002),(24803,248003)];lrs=[a.lr]
    rows=[];scores={lr:[] for lr in lrs}
    for world,init in worlds:
        table=make_world(world)
        for mode in modes:
            val=sample(table,4096,mode,init+101);trainseed=init+211+(0 if mode==modes[0] else 10000)
            for lr in lrs:
                for i,method in enumerate(methods):
                    model,elapsed=fit(method,mode,table,trainseed+i*313+int(lr*10000),lr);m=evaluate(model,val,table)
                    if a.phase=='dev':scores[lr].append(m['token_nll'])
                    ab=address_bits(mode,method);payload=model.serialized_payload_bytes()
                    rows.append({"condition":a.phase,"world_or_seed":world,"mode":mode,"method":method,"serialized_model_bytes":payload,"packet_address_bits":ab,"combined_cost_bytes":f"{payload+ab/8:.3f}","train_tokens_or_examples":UPDATES*64*P,"optimizer_updates":UPDATES,"active_compute_proxy":compute_proxy(P,model.d,UPDATES*64,method),"wall_time_s":round(elapsed,6),"primary_metric":"token_nll","primary_value":f"{m['token_nll']:.10g}","secondary_metric":"joint_packet_accuracy;token_accuracy;valid_path_rate","secondary_value":f"{m['joint_packet_accuracy']:.8g};{m['token_accuracy']:.8g};{m['valid_path_rate']:.8g}","status_note":f"lr={lr}; exact source entropy bits; deterministic config charged"})
                    print(a.phase,world,mode,method,lr,m,"bytes",payload,"secs",round(elapsed,2),flush=True)
    if a.phase=='dev':
        best=min(scores,key=lambda lr:sum(scores[lr])/len(scores[lr]));(ROOT/'DEV_SELECTION.json').write_text(json.dumps({"experiment_id":"MA-248","selected_common_lr":best,"mean_token_nll_by_lr":{str(k):sum(v)/len(v) for k,v in scores.items()},"fresh_worlds":[24801,24802,24803],"updates":UPDATES},indent=2)+'\n')
    with (ROOT/'RESULTS_CORE.csv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n').writerows(rows)
    print(json.dumps({"python":platform.python_version(),"torch":torch.__version__,"threads":torch.get_num_threads(),"rows":len(rows)}))
if __name__=='__main__':main()
