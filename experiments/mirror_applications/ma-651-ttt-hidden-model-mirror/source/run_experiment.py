import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D, RANK, MODES, STREAMS, LENGTH = 16, 4, 4, 16, 128
NOISE_STD, LR = .03, 0.3
SCHEMA = np.frombuffer(b"TTT-LINEAR-STATE-V1", dtype=np.uint8)


def make_world(seed):
    rng = np.random.default_rng(seed)
    base = rng.normal(0, .01, size=(D, D)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    dictionary = q[:, :MODES * RANK].astype(np.float32)
    seqs = []
    for sid in range(STREAMS):
        mode = sid % MODES
        u = dictionary[:, mode * RANK:(mode + 1) * RANK]
        coeff = rng.normal(0, .25, size=(D, RANK))
        target = base.astype(np.float64) + coeff @ u.T
        x = rng.normal(0, 1 / np.sqrt(D), size=(LENGTH, D))
        y = x @ target + rng.normal(0, NOISE_STD, size=(LENGTH, D))
        seqs.append({"mode": mode, "x": x, "y": y})
    return base, dictionary, seqs


def choose_mode(base, dictionary, x, y):
    # Use the first eight self-supervised stream observations to route to a writable subspace.
    err = x @ base - y
    g = (2.0 / len(x)) * x.T @ err
    scores = [np.linalg.norm(g @ dictionary[:, m * RANK:(m + 1) * RANK]) for m in range(MODES)]
    return int(np.argmax(scores)), int(2 * len(x) * D * D + MODES * 2 * D * RANK)


def online_full(base, seq):
    w = base.astype(np.float64).copy(); pred_losses=[]; all_mse=[]; tail_mse=[]
    for t,(x,y) in enumerate(zip(seq["x"],seq["y"])):
        e=x@w-y; mse=float(np.mean(e*e)); all_mse.append(mse)
        if t >= LENGTH-32: tail_mse.append(mse)
        pred_losses.append(.5*(D*np.log(2*np.pi*NOISE_STD**2)+mse/(NOISE_STD**2)))
        w -= LR * 2.0 * np.outer(x,e)
    return w-base, np.mean(pred_losses), np.mean(all_mse), np.mean(tail_mse)


def online_lowrank(base, u, seq):
    b=np.zeros((D,RANK),dtype=np.float64); pred_losses=[]; all_mse=[]; tail_mse=[]
    for t,(x,y) in enumerate(zip(seq["x"],seq["y"])):
        w=base + b@u.T; e=x@w-y; mse=float(np.mean(e*e)); all_mse.append(mse)
        if t >= LENGTH-32: tail_mse.append(mse)
        pred_losses.append(.5*(D*np.log(2*np.pi*NOISE_STD**2)+mse/(NOISE_STD**2)))
        b -= LR * 2.0 * np.outer(x, e@u)
    return b, np.mean(pred_losses), np.mean(all_mse), np.mean(tail_mse)


def save(seed,name,payload):
    p=ROOT/"source"/f"{seed}_{name}.npz";t=time.perf_counter();np.savez_compressed(p,**payload);return p.stat().st_size,time.perf_counter()-t


def run_seed(seed):
    base,dictionary,seqs=make_world(seed);rows=[];state={};seconds={}
    # Full TTT-Linear state bank.
    t=time.perf_counter();deltas=[];losses=[]
    for s in seqs:
        d,nll,mse,tail=online_full(base,s);deltas.append(d.astype(np.float32));losses.append((nll,mse,tail))
    seconds["full_ttt_linear"]=time.perf_counter()-t
    state["full_ttt_linear"]={"schema":SCHEMA,"base_initialization":base,"stream_ids":np.arange(STREAMS,dtype=np.uint8),"writable_deltas":np.stack(deltas)}
    rows.append(("full_ttt_linear",np.mean(losses,axis=0),np.max(np.asarray(losses),axis=0),STREAMS*LENGTH*4*D*D,0))

    # A single fixed basis is the hard-sharing boundary control.
    t=time.perf_counter();fixed_bs=[];fixed_losses=[]
    for s in seqs:
        b,nll,mse,tail=online_lowrank(base,dictionary[:,:RANK],s);fixed_bs.append(b.astype(np.float32));fixed_losses.append((nll,mse,tail))
    seconds["fixed_shared_rank4"]=time.perf_counter()-t
    state["fixed_shared_rank4"]={"schema":SCHEMA,"base_initialization":base,"stream_ids":np.arange(STREAMS,dtype=np.uint8),"shared_basis":dictionary[:,:RANK],"writable_codes":np.stack(fixed_bs)}
    rows.append(("fixed_shared_rank4",np.mean(fixed_losses,axis=0),np.max(fixed_losses,axis=0),STREAMS*LENGTH*6*D*RANK,0))

    # Independent/private output basis upper control.
    t=time.perf_counter();private_bs=[];private_losses=[]
    for s in seqs:
        u=dictionary[:,s["mode"]*RANK:(s["mode"]+1)*RANK];b,nll,mse,tail=online_lowrank(base,u,s);private_bs.append(b.astype(np.float32));private_losses.append((nll,mse,tail))
    seconds["independent_rank4"]=time.perf_counter()-t
    state["independent_rank4"]={"schema":SCHEMA,"base_initialization":base,"stream_ids":np.arange(STREAMS,dtype=np.uint8),"private_bases":np.stack([dictionary[:,s["mode"]*RANK:(s["mode"]+1)*RANK] for s in seqs]),"writable_codes":np.stack(private_bs)}
    rows.append(("independent_rank4",np.mean(private_losses,axis=0),np.max(np.asarray(private_losses),axis=0),STREAMS*LENGTH*6*D*RANK,0))

    # Select an address from the charged global dictionary using initial stream gradients.
    t=time.perf_counter();codes=[];mirror_bs=[];mirror_losses=[];routing_ops=0
    for s in seqs:
        code,ops=choose_mode(base,dictionary,s["x"][:8],s["y"][:8]);routing_ops+=ops;codes.append(code)
        u=dictionary[:,code*RANK:(code+1)*RANK];b,nll,mse,tail=online_lowrank(base,u,s);mirror_bs.append(b.astype(np.float32));mirror_losses.append((nll,mse,tail))
    seconds["mirror_shared_dictionary"]=time.perf_counter()-t
    mirror_payload={"schema":SCHEMA,"base_initialization":base,"stream_ids":np.arange(STREAMS,dtype=np.uint8),"basis_dictionary":dictionary,"stream_codes":np.asarray(codes,dtype=np.uint8),"writable_codes":np.stack(mirror_bs)}
    state["mirror_shared_dictionary"]=mirror_payload
    rows.append(("mirror_shared_dictionary",np.mean(mirror_losses,axis=0),np.max(np.asarray(mirror_losses),axis=0),STREAMS*LENGTH*6*D*RANK,routing_ops))
    native_payload={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in mirror_payload.items()}
    state["native_dictionary_ids"]=native_payload
    rows.append(("native_dictionary_ids",np.mean(mirror_losses,axis=0),np.max(np.asarray(mirror_losses),axis=0),STREAMS*LENGTH*6*D*RANK,routing_ops))
    seconds["native_dictionary_ids"]=seconds["mirror_shared_dictionary"]

    output=[]
    for name,metrics,worst,update_proxy,route_proxy in rows:
        size,serwall=save(seed,name,state[name]);output.append({
            "seed":seed,"method":name,"serialized_state_bytes":size,
            "mean_prequential_nll":float(metrics[0]),"mean_token_mse":float(metrics[1]),"mean_final32_mse":float(metrics[2]),
            "worst_stream_prequential_nll":float(worst[0]),"worst_stream_final32_mse":float(worst[2]),
            "stream_count":STREAMS,"sequence_tokens":STREAMS*LENGTH,"optimizer_updates":STREAMS*LENGTH,
            "update_mac_proxy":int(update_proxy),"router_mac_proxy":int(route_proxy),
            "wall_time_s":seconds[name],"serialization_wall_s":serwall,
        })
    return output


def run(seeds):
    rows=[r for seed in seeds for r in run_seed(seed)]
    (ROOT/"source"/"development.json").write_text(json.dumps(rows,indent=2)+"\n")
    with (ROOT/"RESULTS_CORE.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["condition","world_or_seed","method","serialized_bytes","train_tokens_or_examples","optimizer_updates","active_compute_proxy","wall_time_s","primary_metric","primary_value","secondary_metric","secondary_value","status_note"])
        for r in rows:w.writerow(["four_output_modes",r["seed"],r["method"],r["serialized_state_bytes"],r["sequence_tokens"],r["optimizer_updates"],r["update_mac_proxy"]+r["router_mac_proxy"],r["wall_time_s"],"mean_final32_mse",r["mean_final32_mse"],"mean_prequential_nll",r["mean_prequential_nll"],"development; fresh sealed"])
    return rows


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--seeds",nargs="+",type=int,required=True);args=ap.parse_args();print(json.dumps(run(args.seeds),indent=2))
