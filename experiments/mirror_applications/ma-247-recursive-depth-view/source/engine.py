#!/usr/bin/env python3
"""Development and fresh runner for MA-247."""
from __future__ import annotations
import argparse,csv,json,platform,time
from pathlib import Path
import torch
from model import Config,RecurrentModel,Teacher,mac_proxy_per_example

ROOT=Path(__file__).resolve().parents[1];METHODS=RecurrentModel.METHODS
FIELDS=["condition","world_or_seed","method","serialized_bytes","train_tokens_or_examples","optimizer_updates","active_compute_proxy","wall_time_s","primary_metric","primary_value","secondary_metric","secondary_value","status_note"]

def build_data(cfg,world,seed):
    teacher=Teacher(cfg,world);g=torch.Generator().manual_seed(seed)
    def sample(n):
        x=torch.randn(n,cfg.input_dim,generator=g)
        with torch.no_grad():y=teacher(x)
        return x,y
    return sample(8192),sample(2048),sample(4096)

def mse(model,data):
    model.eval()
    with torch.no_grad():return float((model(data[0])-data[1]).square().mean())

def fit(cfg,method,data,seed,lr,updates):
    torch.manual_seed(seed);model=RecurrentModel(cfg,method);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    x,y=data;g=torch.Generator().manual_seed(seed+29);start=time.perf_counter();model.train()
    for _ in range(updates):
        ix=torch.randint(len(x),(64,),generator=g);pred=model(x[ix]);loss=(pred-y[ix]).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    return model,time.perf_counter()-start

def make_row(phase,world,method,model,elapsed,updates,value,lr,label=None):
    examples=updates*64
    return {"condition":phase,"world_or_seed":world,"method":label or method,"serialized_bytes":model.inference_payload_bytes(),
      "train_tokens_or_examples":examples,"optimizer_updates":updates,"active_compute_proxy":mac_proxy_per_example(model.cfg,method)*examples*3,
      "wall_time_s":round(elapsed,6),"primary_metric":"final_state_mse","primary_value":f"{value:.10g}",
      "secondary_metric":"parameter_count","secondary_value":str(model.parameter_count()),"status_note":f"common_lr={lr}; residual recurrent gate in all methods"}

def append(rows):
    p=ROOT/"RESULTS_CORE.csv"
    with p.open("a",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator="\n")
        if f.tell()==0:w.writeheader()
        w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=("dev","fresh"),required=True);ap.add_argument("--lr",type=float);args=ap.parse_args()
    torch.set_num_threads(1);cfg=Config();updates=1200
    if args.phase=="dev":worlds=[(24700,247000)];lrs=[0.003,0.01]
    else:
        if args.lr not in (0.003,0.01):raise SystemExit("fresh requires frozen --lr 0.003 or 0.01")
        worlds=[(24701,247001),(24702,247002),(24703,247003)];lrs=[args.lr]
    rows=[];scores={lr:[] for lr in lrs}
    for world,seed in worlds:
        train,dev,test=build_data(cfg,world,seed)
        for lr in lrs:
            for i,method in enumerate(METHODS):
                model,elapsed=fit(cfg,method,train,seed+i*181+int(lr*10000),lr,updates);dv=mse(model,dev)
                if args.phase=="dev":
                    scores[lr].append(dv);rows.append(make_row("dev",world,method,model,elapsed,updates,dv,lr,label=method+f"_lr{lr:g}"));tv=None
                else:
                    tv=mse(model,test);rows.append(make_row("fresh",world,method,model,elapsed,updates,tv,lr))
                msg=f"{args.phase} world={world} method={method} lr={lr} dev={dv:.8g}"
                if tv is not None:msg+=f" test={tv:.8g}"
                print(msg+f" bytes={model.inference_payload_bytes()} wall={elapsed:.2f}s",flush=True)
    if args.phase=="dev":
        best=min(scores,key=lambda lr:sum(scores[lr])/len(scores[lr]))
        out={"experiment_id":"MA-247","selected_common_lr":best,"dev_mse_by_lr":{str(k):v for k,v in scores.items()},"fresh_worlds":[24701,24702,24703],"updates":updates,"source_freeze_stage":"before fresh evaluation"}
        (ROOT/"DEV_SELECTION.json").write_text(json.dumps(out,indent=2)+"\n");print("selected_common_lr=",best)
    append(rows);print(json.dumps({"python":platform.python_version(),"torch":torch.__version__,"threads":torch.get_num_threads(),"rows":len(rows)}))

if __name__=="__main__":main()
