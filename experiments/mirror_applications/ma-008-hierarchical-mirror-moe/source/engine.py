from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from model import Config, METHODS, MODES, HierarchicalMoE, Teacher, active_mac_proxy, coordinate_flop_proxy, router_mac_proxy


ROLE_CENTERS = torch.tensor([[-1.4, 1.4], [-1.4, -1.4], [1.4, 1.4], [1.4, -1.4]])


def make_balanced(seed: int, n: int, cfg: Config):
    bsz = cfg.experts * 16
    if n % bsz: raise ValueError("balanced examples must be a whole number of protocol batches")
    labels = torch.arange(cfg.experts).repeat_interleave(16).repeat(n // bsz)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((n, cfg.input_dim), generator=gen)
    x[:, :2] += ROLE_CENTERS[labels]
    return x, labels


def eval_model(model, x, labels, target):
    model.eval()
    with torch.no_grad():
        pred, role, primary, child = model(x)
        role_acc = float((role == labels).float().mean().item())
        group_pred = (primary.argmax(-1) if model.hierarchical else role // model.cfg.experts_per_group)
        group_acc = float((group_pred == labels // model.cfg.experts_per_group).float().mean().item())
        per_role = []
        for e in range(model.cfg.experts):
            mask = labels == e
            per_role.append(float(F.mse_loss(pred[mask], target[mask]).item()))
        return {
            "heldout_mse": float(F.mse_loss(pred, target).item()),
            "role_accuracy": role_acc,
            "group_accuracy": group_acc,
            "token_coverage": 1.0,
            "per_role_mse_json": json.dumps(per_role, separators=(",", ":")),
            "router_mac_proxy_per_example": router_mac_proxy(model.cfg, model.method),
            "active_mac_proxy_per_example": active_mac_proxy(model.cfg, model.method),
            "coordinate_flop_proxy_per_example": coordinate_flop_proxy(model.cfg, model.method),
        }


def train_one(world, teacher_seed, init_seed, mode, method, lr, updates, batch_size, inference_repetitions):
    cfg = Config()
    torch.set_num_threads(1)
    teacher = Teacher(cfg, mode, teacher_seed)
    train_x, train_labels = make_balanced(100_000_000 + world, updates*batch_size, cfg)
    val_x, val_labels = make_balanced(110_000_000 + world, 2048, cfg)
    with torch.no_grad():
        train_y = teacher.forward(train_x, train_labels)
        val_y = teacher.forward(val_x, val_labels)
    torch.manual_seed(init_seed)
    model = HierarchicalMoE(cfg, method)
    router_params = [p for n,p in model.named_parameters() if "router" in n]
    other_params = [p for n,p in model.named_parameters() if "router" not in n]
    opt = torch.optim.AdamW([{"params":router_params,"weight_decay":0.0},{"params":other_params,"weight_decay":1e-4}],lr=lr)
    start = time.perf_counter()
    for step in range(updates):
        lo = step*batch_size
        xb=train_x[lo:lo+batch_size]; yb=train_y[lo:lo+batch_size]; labels=train_labels[lo:lo+batch_size]
        model.train()
        pred, role, primary, child = model(xb)
        loss=F.mse_loss(pred,yb)+model.router_loss(primary,child,labels)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    wall=time.perf_counter()-start
    metrics=eval_model(model,val_x,val_labels,val_y)
    infer_x,_=make_balanced(120_000_000+world,1024,cfg)
    model.eval()
    with torch.no_grad():
        for _ in range(3): model(infer_x)
        t0=time.perf_counter()
        for _ in range(inference_repetitions): model(infer_x)
        infer_sec=time.perf_counter()-t0
    payload=model.serialize()
    loaded=torch.load(__import__('io').BytesIO(payload),map_location='cpu',weights_only=False)
    clone=HierarchicalMoE(cfg,method); clone.load_state_dict(loaded['state_dict'])
    with torch.no_grad():
        if not torch.equal(clone(val_x)[0],model(val_x)[0]): raise RuntimeError('serialized inference mismatch')
    return {'world':world,'teacher_seed':teacher_seed,'initialization_seed':init_seed,'teacher_mode':mode,'method':method,'learning_rate':lr,'updates':updates,'batch_size':batch_size,'examples_seen':updates*batch_size,**metrics,'serialized_bytes':len(payload),'training_wall_seconds':wall,'inference_batch_size':infer_x.shape[0],'inference_repetitions':inference_repetitions,'inference_examples_per_second':infer_x.shape[0]*inference_repetitions/infer_sec,'parameter_count_diagnostic':sum(p.numel() for p in model.parameters())}


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=('dev','fresh'),required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--selection',type=Path); ap.add_argument('--updates',type=int,default=1200); ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--inference-repetitions',type=int,default=20); args=ap.parse_args()
    if args.stage=='dev': runs=[(80000,800000,8000000,lr) for lr in (0.001,0.003,0.01)]
    else:
        if args.selection is None: ap.error('--selection required for fresh')
        lr=float(json.loads(args.selection.read_text())['selected_learning_rate']); runs=[(w,800000+w-80000,8000000+w-80000,lr) for w in (80001,80002,80003)]
    rows=[]
    for world,ts,is_,lr in runs:
        for mode in MODES:
            for i,m in enumerate(METHODS):
                row=train_one(world,ts,is_+i*100,mode,m,lr,args.updates,args.batch_size,args.inference_repetitions)
                print(f"{world} {mode} {m} lr={lr:g} mse={row['heldout_mse']:.7g} bytes={row['serialized_bytes']} route={row['role_accuracy']:.3f} t={row['training_wall_seconds']:.2f}s",flush=True); rows.append(row)
    write_csv(args.output,rows)


if __name__=='__main__': main()
