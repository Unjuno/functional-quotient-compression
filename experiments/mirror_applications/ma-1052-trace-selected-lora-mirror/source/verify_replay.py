"""Reconstruct serialized MA-1052 inference states and replay recorded metrics."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

from run_experiment import ROOT, ROLES, eval_model, load_model, load_series, role_data

def load_adapter_state(model, adapters, directory, selected, role_codes):
    state=load_file(str(directory/"adapter.safetensors"),device="cpu")
    for n,m in adapters.items():
        m.selected.fill_(n in selected)
        if n not in selected: continue
        m.a.weight.data.copy_(state[n+".A"])
        m.b.weight.data.copy_(state[n+".B"])
        m.gate.data.copy_(state[n+".gate"].reshape_as(m.gate))
        if role_codes and n+".view_code" in state: m.code.data.copy_(state[n+".view_code"])
    head={k[len("head."):]:v for k,v in state.items() if k.startswith("head.")}
    model.head.load_state_dict(head)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--run",required=True); a=ap.parse_args()
    torch.set_num_threads(4)
    run_path=Path(a.run)/"run.json"; recorded=json.loads(run_path.read_text()); split=recorded["evaluation"]
    series,splits=load_series(); data={r:role_data(series,splits,r,split) for r in ROLES}
    replay={}
    for name,view in (("pooled_trace_shared","shared"),("pooled_trace_diag","diag"),("pooled_trace_mirror","mirror")):
        info=recorded["methods"][name]; selected=info["selected"]
        model,adapters=load_model(a.checkpoint,view,720)
        load_adapter_state(model,adapters,Path(a.run)/name,selected,view in ("diag","mirror"))
        replay[name]=eval_model(model,adapters,ROLES,data,split)
        del model,adapters
    specialist={}
    for role in ROLES:
        name=f"specialist_{role[0]}_H{role[1]}"; info=recorded["methods"][name]
        model,adapters=load_model(a.checkpoint,"shared",role[1],roles=1)
        load_adapter_state(model,adapters,Path(a.run)/name,info["selected"],False)
        role_metrics=eval_model(model,adapters,[role],{role:data[role]},split)
        replay[name]=role_metrics
        specialist.update(role_metrics)
        del model,adapters
    replay["independent_trace_reference"]={"metrics":specialist,"macro_mase":float(np.mean([x["mase"] for x in specialist.values() if isinstance(x,dict)]))}
    for role in ROLES:
        name=f"moment_linear_probe_{role[0]}_H{role[1]}"
        model,_=load_model(a.checkpoint,"shared",role[1],roles=1,linear_probe=True)
        state=load_file(str(Path(a.run)/name/"head.safetensors"),device="cpu")
        model.head.load_state_dict(state)
        replay[name]=eval_model(model,{},[role],{role:data[role]},split)
        del model
    diffs={}
    for name,metrics in replay.items():
        orig=recorded["methods"][name].get("metrics",{})
        if name=="independent_trace_reference":
            diffs[name+".macro_mase"]=abs(metrics["macro_mase"]-recorded["methods"][name]["macro_mase"])
            continue
        if "macro_mase" in metrics and "macro_mase" in orig:
            diffs[name+".macro_mase"]=abs(metrics["macro_mase"]-orig["macro_mase"])
        for key,value in metrics.items():
            if isinstance(value,dict) and key in orig:
                diffs[name+"."+key+".mase"]=abs(value["mase"]-orig[key]["mase"])
    payload_path=Path(a.run)/"payload_metrics.json"
    canonicalized=not payload_path.exists()
    if canonicalized:
        payload_path.write_text(json.dumps(replay,indent=2)+"\n")
        payload_diffs={}
    else:
        expected=json.loads(payload_path.read_text())
        payload_diffs={}
        for name,metrics in replay.items():
            old=expected[name]
            if name=="independent_trace_reference":
                payload_diffs[name+".macro_mase"]=abs(metrics["macro_mase"]-old["macro_mase"])
                continue
            for key,value in metrics.items():
                if isinstance(value,dict) and key in old:
                    payload_diffs[name+"."+key+".mase"]=abs(value["mase"]-old[key]["mase"])
                elif isinstance(value,(float,int)) and key in old:
                    payload_diffs[name+"."+key]=abs(value-old[key])
    in_memory_max=max(diffs.values(),default=0.0)
    payload_max=max(payload_diffs.values(),default=0.0)
    result={"evaluation":split,"replayed_methods":len(replay),
            "canonical_payload_metrics_created":canonicalized,
            "payload_replay_max_abs_difference":payload_max,
            "payload_replay_exact":(canonicalized or all(v<1e-7 for v in payload_diffs.values())),
            "in_memory_to_serialized_payload_max_abs_difference":in_memory_max,
            "in_memory_to_payload_differences":diffs,
            "payload_replay_differences":payload_diffs}
    result["passed"]=result["payload_replay_exact"]
    dest=Path(a.run)/"replay_verification.json"; dest.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))
    if not result["passed"]: raise SystemExit(1)

if __name__=="__main__": main()
