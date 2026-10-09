#!/usr/bin/env python3
"""MA-527 shared SAE feature bank with a deterministic Givens View."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys, time, zipfile
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]
ART = ROOT / "artifacts/ma527"
HARNESS = Path(__file__).with_name("shared_fv_harness.py")
spec = importlib.util.spec_from_file_location("ma527_shared_harness", HARNESS)
h = importlib.util.module_from_spec(spec); sys.modules[spec.name] = h; spec.loader.exec_module(h)

SAE_REPO = "Elriggs/pythia-70M-deduped-sae"
SAE_REV = "36c2027509efad69836aa4231999c9b717848ecd"
SAE_SHA = "85b57dfe34d19769e2df0e208e38fda8815da8493c56b703a99ff74cc610dbf4"
FIT = np.arange(12); DEV = (12, 13); FRESH = (14, 15)
RANK = 8; BANK = 32
ALPHAS = (0.25, 0.5, 1.0); ANGLE_SCALES = (0.05, 0.1, 0.2, 0.4)
torch.set_num_threads(5)

class TiedSAE(torch.nn.Module):
    """Safe module shell for the pinned checkpoint's tied-SAE parameters."""
    def __init__(self): super().__init__()

def sha(path):
    digest=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): digest.update(block)
    return digest.hexdigest()

def load_sae(path):
    if sha(path) != SAE_SHA: raise ValueError("pinned SAE hash mismatch")
    torch.serialization.add_safe_globals([TiedSAE])
    obj=torch.load(path,map_location="cpu",weights_only=True)
    W=obj.encoder.detach().cpu().numpy().astype(np.float32)
    bias=obj.encoder_bias.detach().cpu().numpy().astype(np.float32)
    if W.shape!=(2048,512) or bias.shape!=(2048,): raise ValueError("unexpected SAE tensor shape")
    return W,bias

def encode(vectors,W,bias): return np.maximum(vectors@W.T+bias,0).astype(np.float32)

def choose_bank(z):
    return np.argsort(-z[FIT].sum(axis=0),kind="stable")[:BANK].astype(np.int16)

def givens_matrix(scale):
    angles=np.asarray([scale*math.sin(2*math.pi*(j+1)/17) for j in range(16)],dtype=np.float64)
    G=np.eye(BANK,dtype=np.float64)
    for j,theta in enumerate(angles):
        c,s=math.cos(theta),math.sin(theta); i=2*j
        G[[i,i+1],:]=np.array([[c,-s],[s,c]])@G[[i,i+1],:]
    return G.astype(np.float32),angles.astype(np.float32)

def sparse_topk(z,k=RANK):
    ids=np.argsort(-np.abs(z),axis=1,kind="stable")[:,:k]
    vals=np.take_along_axis(z,ids,axis=1)
    mask=np.take_along_axis(z!=0,ids,axis=1)
    vals=np.where(mask,vals,0).astype(np.float32)
    return ids.astype(np.uint8),vals

def decode_codes(ids,vals,atoms):
    out=np.zeros((len(ids),atoms.shape[1]),dtype=np.float32)
    for r in range(len(ids)):
        out[r]=vals[r]@atoms[ids[r]]
    return out

def omp(vectors,atoms,k=RANK):
    norms=np.linalg.norm(atoms.astype(np.float64),axis=1)
    unit=(atoms/norms[:,None]).astype(np.float32)
    ids=np.zeros((len(vectors),k),dtype=np.uint16);vals=np.zeros((len(vectors),k),dtype=np.float32)
    dec=np.zeros_like(vectors,dtype=np.float32)
    for r,v in enumerate(vectors):
        res=v.copy();used=np.zeros(len(atoms),dtype=bool)
        for j in range(k):
            corr=unit@res;corr[used]=0;ix=int(np.argmax(np.abs(corr)))
            if abs(float(corr[ix]))<1e-12: break
            coef=float(corr[ix]/norms[ix]);ids[r,j]=ix;vals[r,j]=coef;used[ix]=True
            dec[r]+=coef*atoms[ix];res-=coef*atoms[ix]
    return ids,vals,dec

def payload(path,method,metadata,parts):
    np.savez(path,**metadata,**parts)
    with zipfile.ZipFile(path) as z:
        if any(i.compress_type!=zipfile.ZIP_STORED for i in z.infolist()): raise AssertionError("NPZ must be uncompressed")
    return path.stat().st_size

def method_codes(vectors,W,bias,angle_scale):
    z=encode(vectors,W,bias);bank=choose_bank(z);atoms=W[bank]
    native_ids,native_vals=sparse_topk(z)
    native_dec=decode_codes(native_ids.astype(np.int64),native_vals,W)
    bank_z=z[:,bank];base_ids,base_vals=sparse_topk(bank_z)
    base_dec=decode_codes(base_ids.astype(np.int64),base_vals,atoms)
    G,angles=givens_matrix(angle_scale)
    transformed=bank_z@G.T
    mir_ids,mir_vals=sparse_topk(transformed)
    mir_dec=decode_codes(mir_ids.astype(np.int64),mir_vals,atoms)
    global_ids,global_vals,global_dec=omp(vectors,W)
    # Shared low-rank / LoReFT control: rank-eight basis fit only on fit-task FVs.
    mean,basis,codes,loreft_dec=h.pca_fit(vectors,8)
    return bank,angles,{
        "native_sae_top8":(native_dec,{"feature_ids":native_ids,"coefficients":native_vals}),
        "shared_bank_sparse":(base_dec,{"bank_feature_ids":bank,"local_ids":base_ids,"coefficients":base_vals}),
        "mirror_givens":(mir_dec,{"bank_feature_ids":bank,"local_ids":mir_ids,"coefficients":mir_vals,"givens_angles":angles.astype(np.float16)}),
        "global_omp8":(global_dec,{"feature_ids":global_ids,"coefficients":global_vals}),
        "loreft_rank8":(loreft_dec,{"mean":mean,"basis":basis,"task_codes":codes.astype(np.float32)}),
    }

def run_seed(seed,phase,model_dir,sae_path,angle_scale=None,alpha=None):
    model,tok,torchmod=h.load_model(model_dir)
    t0=time.perf_counter();vectors,manifest,support=h.extract_task_vectors(model,tok,torchmod,seed);extract_s=time.perf_counter()-t0
    W,bias=load_sae(sae_path);bank,angles,methods=method_codes(vectors,W,bias,angle_scale or 0.05)
    task_ids=DEV if phase=="dev" else FRESH
    results=[];t_eval=time.perf_counter()
    for name,(decoded,parts) in methods.items():
        scale=float(alpha or 1.0);tm=time.perf_counter();metrics=h.evaluate(model,tok,torchmod,vectors,decoded*scale,manifest,task_ids=task_ids);method_s=time.perf_counter()-tm
        results.append({"method":name,"alpha":scale,"metrics":metrics,"decoded":decoded,"parts":parts,"wall_method_seconds":method_s})
    te=time.perf_counter();explicit_metrics=h.evaluate(model,tok,torchmod,vectors,vectors*float(alpha or 1.0),manifest,task_ids=task_ids);explicit_s=time.perf_counter()-te
    tn=time.perf_counter();none_metrics=h.evaluate(model,tok,torchmod,vectors,np.zeros_like(vectors),manifest,task_ids=task_ids);none_s=time.perf_counter()-tn
    eval_s=time.perf_counter()-t_eval
    outdir=ART/phase/f"seed_{seed}";outdir.mkdir(parents=True,exist_ok=True)
    basefiles=["model.safetensors","config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json"]
    model_bytes=sum((Path(model_dir)/n).stat().st_size for n in basefiles if (Path(model_dir)/n).exists())
    metadata={"task_ids":np.arange(16,dtype=np.int16),"model_sha256":np.frombuffer(h.MODEL_SHA.encode(),dtype="S64"),
      "sae_sha256":np.frombuffer(SAE_SHA.encode(),dtype="S64"),"layer":np.array([h.HOOK_LAYER],np.int16),"schema":np.array([527,BANK,RANK],np.int32)}
    explicit_path=outdir/"explicit_fv.npz";explicit_bytes=payload(explicit_path,"explicit",metadata,{"function_vectors":vectors.astype(np.float32),"intervention_scale":np.asarray([float(alpha or 1.0)],dtype=np.float32)})
    rows=[]
    for row in results:
        name=row["method"];parts=row["parts"].copy();parts.update(metadata)
        # alpha is reconstruction state and must be serialized.
        parts["intervention_scale"]=np.asarray([row["alpha"]],dtype=np.float32)
        p=outdir/f"{name}.npz";nbytes=payload(p,name,{},parts)
        uses_sae=name!="loreft_rank8";common=model_bytes+(Path(sae_path).stat().st_size if uses_sae else 0)
        rowout={k:v for k,v in row.items() if k not in ("decoded","parts")}
        rowout.update({"seed":seed,"payload_bytes":nbytes,"model_bytes":model_bytes,"sae_bytes":Path(sae_path).stat().st_size if uses_sae else 0,
          "standalone_total_bytes":common+nbytes,"explicit_payload_bytes":explicit_bytes,"shared_base_bytes":common,
          "nonzeros_per_task":int(np.count_nonzero(row["parts"]["coefficients"] if "coefficients" in row["parts"] else row["parts"].get("task_codes",[]))/16) if name!="loreft_rank8" else 8,
          "wall_eval_seconds":row["wall_method_seconds"],"candidate_input_tokens":row["metrics"]["candidate_input_tokens"],"candidate_sequences":row["metrics"]["candidate_sequences"],
          "support_examples":128,"support_input_tokens":support["input_tokens"],"optimizer_updates":0,"active_compute_proxy":row["metrics"]["candidate_sequences"]*512+16*BANK*RANK})
        rows.append(rowout)
    report={"experiment_id":"MA-527","phase":phase,"seed":seed,"angle_scale":angle_scale,"alpha":alpha,
      "bank_feature_ids":bank.tolist(),"angles_radians":angles.tolist(),"train_fit_task_ids":FIT.tolist(),"evaluated_task_ids":list(task_ids),
      "support_seconds":extract_s,"evaluation_seconds":eval_s,"explicit_reference":explicit_metrics,"explicit_wall_seconds":explicit_s,"no_intervention":none_metrics,"no_intervention_wall_seconds":none_s,
      "rows":rows,"model_bytes":model_bytes,"sae_bytes":Path(sae_path).stat().st_size}
    (outdir/"metrics.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=["dev","fresh"],required=True);ap.add_argument("--seeds",nargs="+",type=int,required=True)
    ap.add_argument("--angle-scale",type=float);ap.add_argument("--alpha",type=float);a=ap.parse_args()
    ART.mkdir(parents=True,exist_ok=True)
    model_dir=snapshot_download("EleutherAI/pythia-70m-deduped",revision="e93a9faa9c77e5d09219f6c868bfc7a1bd65593c",allow_patterns=["model.safetensors","config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json"])
    sae_path=hf_hub_download(SAE_REPO,"pythia-70m-deduped_r4_gpt_neox.layers.3.pt",revision=SAE_REV)
    freeze_path=EXP/"DEV_FREEZE.json"
    if a.phase=="fresh":
        if not freeze_path.exists(): raise RuntimeError("fresh is sealed until DEV_FREEZE.json exists")
        fr=json.loads(freeze_path.read_text())
        if fr.get("decision")!="OPEN_FRESH": raise RuntimeError("development gates failed; fresh must remain sealed")
        a.angle_scale=float(fr["givens_pattern_scale"]);a.alpha=float(fr["intervention_scale"])
        if sorted(a.seeds)!=[52711,52712,52713]: raise ValueError("fresh seeds differ from frozen protocol")
    else:
        if (a.angle_scale not in (None,0.1)) or (a.alpha not in (None,0.5)): raise ValueError("development configuration is frozen at angle_scale=0.1, alpha=0.5")
        if sorted(a.seeds)!=[52701,52702]: raise ValueError("development seeds differ from frozen protocol")
        a.angle_scale=0.1;a.alpha=0.5
    reports=[]
    for seed in a.seeds:
        rep=run_seed(seed,a.phase,model_dir,sae_path,a.angle_scale,a.alpha);reports.append(rep)
        print(json.dumps({"phase":a.phase,"seed":seed,"explicit":rep["explicit_reference"],"methods":[{"name":r["method"],"metrics":r["metrics"],"bytes":r["payload_bytes"]} for r in rep["rows"]]},indent=2))
    (ART/a.phase/f"summary.json").write_text(json.dumps(reports,indent=2,sort_keys=True)+"\n")
    if a.phase=="dev":
        ok=True; checks=[]
        for report in reports:
            ex=report["explicit_reference"]
            m=next(x for x in report["rows"] if x["method"]=="mirror_givens")
            delta=ex["mean_gold_candidate_logprob"]-m["metrics"]["mean_gold_candidate_logprob"]
            acc_gap=ex["heldout_accuracy"]-m["metrics"]["heldout_accuracy"]
            byte_gate=m["payload_bytes"]<=0.5*m["explicit_payload_bytes"]
            qgate=delta<=0.10 and acc_gap<=0.05
            controls=[x for x in report["rows"] if x["method"] in ("shared_bank_sparse","loreft_rank8")]
            dominated=any(x["payload_bytes"]<=1.1*m["payload_bytes"] and
              x["metrics"]["mean_gold_candidate_logprob"]>=m["metrics"]["mean_gold_candidate_logprob"] and
              x["metrics"]["heldout_accuracy"]>=m["metrics"]["heldout_accuracy"] for x in controls)
            passed=bool(byte_gate and qgate and m["nonzeros_per_task"]<=16 and not dominated);ok &= passed
            checks.append({"seed":report["seed"],"gold_logprob_loss_nats":delta,"accuracy_loss":acc_gap,"mirror_bytes":m["payload_bytes"],"explicit_bytes":m["explicit_payload_bytes"],"quality_gate":qgate,"byte_gate":byte_gate,"dominated_by_simple_control":dominated,"passed":passed})
        freeze={"experiment_id":"MA-527","decision":"OPEN_FRESH" if ok else "FAIL_DEV_NO_FRESH_METRICS",
          "givens_pattern_scale":0.1,"intervention_scale":0.5,"dev_checks":checks,
          "dev_summary_sha256":sha(ART/"dev/summary.json"),"fresh_seeds":[52711,52712,52713],
          "note":"No parameter selection: one fixed configuration. Inherited extraction computes codes for task IDs 0-15; when the dev gate fails, do not evaluate fresh seeds.",
          "fresh_task_identity_exposure":"Task IDs 14-15 are represented in dev-seed extraction artifacts; no fresh seeds or fresh causal evaluations were run."}
        freeze_path.write_text(json.dumps(freeze,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"development_decision":freeze["decision"],"checks":checks},indent=2))
if __name__=="__main__": main()
