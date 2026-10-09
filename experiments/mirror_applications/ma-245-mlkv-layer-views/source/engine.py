#!/usr/bin/env python3
"""Development/fresh runner for MA-245."""
from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import torch
from model import Config,LayerKV,Teacher,mac_proxy_per_example

ROOT=Path(__file__).resolve().parents[1]
METHODS=LayerKV.METHODS
FIELDS=["condition","world_or_seed","method","serialized_bytes","train_tokens_or_examples","optimizer_updates","active_compute_proxy","wall_time_s","primary_metric","primary_value","secondary_metric","secondary_value","status_note"]

def build_data(cfg,world,seed):
    teacher=Teacher(cfg,world);g=torch.Generator().manual_seed(seed)
    def sample(n):
        query=torch.randn(n,cfg.layers,cfg.head_dim,generator=g)
        mem=torch.randn(n,cfg.memory_tokens,cfg.model_dim,generator=g)
        with torch.no_grad():target=teacher(query,mem)
        return query,mem,target
    return sample(8192),sample(2048),sample(4096)

def mse(model,data):
    model.eval()
    with torch.no_grad():return float((model(data[0],data[1])-data[2]).square().mean())

def fit(cfg,method,data,seed,lr,updates):
    torch.manual_seed(seed);model=LayerKV(cfg,method);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    q,mem,y=data;g=torch.Generator().manual_seed(seed+13);model.train();start=time.perf_counter()
    for _ in range(updates):
        ix=torch.randint(len(q),(64,),generator=g);pred=model(q[ix],mem[ix]);loss=(pred-y[ix]).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return model,time.perf_counter()-start

def make_row(phase,world,method,model,elapsed,updates,value,lr,label=None):
    examples=updates*64
    return {"condition":phase,"world_or_seed":world,"method":label or method,"serialized_bytes":model.inference_payload_bytes(),
      "train_tokens_or_examples":examples,"optimizer_updates":updates,"active_compute_proxy":mac_proxy_per_example(model.cfg,method)*examples*3,
      "wall_time_s":round(elapsed,6),"primary_metric":"layer_attention_output_mse","primary_value":f"{value:.10g}",
      "secondary_metric":"kv_cache_bytes_8_tokens","secondary_value":str(model.cache_bytes()),"status_note":f"common_lr={lr}; same memory input supplied to every layer"}

def append(rows):
    p=ROOT/"RESULTS_CORE.csv"
    with p.open("a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator="\n")
        if f.tell()==0:w.writeheader()
        w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=("dev","fresh"),required=True);ap.add_argument("--lr",type=float);args=ap.parse_args()
    torch.set_num_threads(1);cfg=Config();updates=900
    if args.phase=="dev":worlds=[(24500,245000)];lrs=[0.003,0.01]
    else:
        if args.lr not in (0.003,0.01):raise SystemExit("fresh requires frozen --lr 0.003 or 0.01")
        worlds=[(24501,245001),(24502,245002),(24503,245003)];lrs=[args.lr]
    rows=[];scores={lr:[] for lr in lrs}
    for world,seed in worlds:
        train,dev,test=build_data(cfg,world,seed)
        for lr in lrs:
            for i,method in enumerate(METHODS):
                model,elapsed=fit(cfg,method,train,seed+i*179+int(lr*10000),lr,updates);dv=mse(model,dev)
                if args.phase=="dev":
                    scores[lr].append(dv);rows.append(make_row("dev",world,method,model,elapsed,updates,dv,lr,label=method+f"_lr{lr:g}"));tv=None
                else:
                    tv=mse(model,test);rows.append(make_row("fresh",world,method,model,elapsed,updates,tv,lr))
                msg=f"{args.phase} world={world} method={method} lr={lr} dev={dv:.8g}"
                if tv is not None:msg+=f" test={tv:.8g}"
                print(msg+f" bytes={model.inference_payload_bytes()} cache={model.cache_bytes()} wall={elapsed:.2f}s",flush=True)
    if args.phase=="dev":
        selected=min(scores,key=lambda lr:sum(scores[lr])/len(scores[lr]))
        f={"experiment_id":"MA-245","selected_common_lr":selected,"dev_mse_by_lr":{str(k):v for k,v in scores.items()},"fresh_worlds":[24501,24502,24503],"updates":updates,"source_freeze_stage":"before fresh evaluation"}
        (ROOT/"DEV_SELECTION.json").write_text(json.dumps(f,indent=2)+"\n");print("selected_common_lr=",selected)
    append(rows);print(json.dumps({"python":platform.python_version(),"torch":torch.__version__,"threads":torch.get_num_threads(),"rows":len(rows)}))

if __name__=="__main__":main()
