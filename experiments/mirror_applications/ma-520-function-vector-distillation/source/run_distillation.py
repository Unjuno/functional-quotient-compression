#!/usr/bin/env python3
"""Run frozen MA-520 learned shared linear FV decoder screen."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time, zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path(__file__).with_name('shared_fv_harness.py')
spec = importlib.util.spec_from_file_location('ma516_shared_fv_harness', HARNESS)
h = importlib.util.module_from_spec(spec); sys.modules[spec.name] = h; spec.loader.exec_module(h)
RANK = 4
FIT_TASKS = np.arange(12)
HELD_TASKS = np.arange(12, 16)
MODEL_HASH = h.MODEL_SHA
REVISION = h.REVISION

def learned_linear_fit(vectors: np.ndarray, seed: int, updates: int = 2000):
    """Full-batch Adam fit of mean + Wm against the first twelve FV teachers."""
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    train = vectors[FIT_TASKS].astype(np.float32)
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    gen = torch.Generator(device='cpu'); gen.manual_seed(seed)
    w = torch.nn.Parameter(torch.randn((512, RANK), generator=gen) / np.sqrt(512.0))
    codes = torch.nn.Parameter(torch.randn((12, RANK), generator=gen) / np.sqrt(512.0))
    target = torch.as_tensor(train - mean)
    optimizer = torch.optim.Adam([w, codes], lr=0.01, weight_decay=0.0)
    start = time.perf_counter()
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        loss = ((codes @ w.T - target) ** 2).mean()
        loss.backward(); optimizer.step()
    fit_seconds = time.perf_counter() - start
    basis = w.detach().cpu().numpy().astype(np.float32)
    learned_codes = np.zeros((16, RANK), dtype=np.float32)
    learned_codes[FIT_TASKS] = codes.detach().cpu().numpy().astype(np.float32)
    assign_start = time.perf_counter()
    held_codes = np.linalg.lstsq(basis.astype(np.float64),
                                 (vectors[HELD_TASKS] - mean).astype(np.float64).T,
                                 rcond=None)[0].T.astype(np.float32)
    learned_codes[HELD_TASKS] = held_codes
    assignment_seconds = time.perf_counter() - assign_start
    return mean, basis, learned_codes, fit_seconds, assignment_seconds

def native_pca(vectors: np.ndarray):
    train = vectors[FIT_TASKS].astype(np.float64)
    mean = train.mean(axis=0).astype(np.float32)
    _, _, vt = np.linalg.svd(train - mean, full_matrices=False)
    basis = vt[:RANK].T.astype(np.float32)
    codes = np.zeros((16, RANK), dtype=np.float32)
    codes[:] = (vectors - mean) @ basis
    return mean, basis, codes

def quantize_per_vector(vectors: np.ndarray):
    scales = np.max(np.abs(vectors), axis=1).astype(np.float32) / 127.0
    scales = np.maximum(scales, np.finfo(np.float32).tiny)
    q = np.clip(np.rint(vectors / scales[:, None]), -127, 127).astype(np.int8)
    return q, scales, q.astype(np.float32) * scales[:, None]

def payload(path: Path, method: str, vectors: np.ndarray, parts=None):
    common = dict(task_ids=np.arange(16, dtype=np.int16), layer=np.array([h.LAYER_OUT], np.int16),
                  model_revision=np.frombuffer(REVISION.encode(), dtype='S40'),
                  model_sha256=np.frombuffer(MODEL_HASH.encode(), dtype='S64'),
                  schema=np.array([520, 1, RANK], dtype=np.int32))
    if method == 'none': return 0
    if method == 'explicit_fp32': arrays = dict(function_vectors=vectors.astype(np.float32), **common)
    elif method == 'explicit_int8': arrays = dict(quantized_vectors=parts[0], scales=parts[1], **common)
    else:
        mean, basis, codes = parts
        arrays = dict(shared_mean=mean, shared_decoder=basis, function_codes=codes, **common)
    np.savez(path, **arrays)
    size = path.stat().st_size
    with zipfile.ZipFile(path) as z:
        if any(item.compress_type != zipfile.ZIP_STORED for item in z.infolist()):
            raise AssertionError('payload must be uncompressed NPZ')
    return size

def vector_metrics(reference: np.ndarray, decoded: np.ndarray):
    x = reference[HELD_TASKS].astype(np.float64); y = decoded[HELD_TASKS].astype(np.float64)
    err = x-y
    cosine = np.sum(x*y, axis=1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)+1e-12)
    return {'heldout_relative_rmse':float(np.linalg.norm(err)/np.linalg.norm(x)),
            'heldout_cosine_mean':float(np.mean(cosine)),
            'heldout_cosine_min':float(np.min(cosine))}

def run(seed: int, model_dir: str, out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    model, tok, torch = h.load_model(model_dir)
    t0=time.perf_counter(); vectors, manifest, support = h.extract_task_vectors(model,tok,torch,seed)
    extraction_s=time.perf_counter()-t0
    np.savez(out/'teacher_vectors.npz',task_vectors=vectors,task_ids=np.arange(16,dtype=np.int16))
    (out/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    common_names=['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']
    common_bytes=sum((Path(model_dir)/n).stat().st_size for n in common_names)
    fit_start=time.perf_counter(); pca_mean,pca_basis,pca_codes=native_pca(vectors);pca_fit_s=time.perf_counter()-fit_start
    learned_mean,learned_basis,learned_codes,learned_fit_s,learned_assign_s=learned_linear_fit(vectors,seed)
    q,scales,int8_decoded=quantize_per_vector(vectors)
    methods={
      'none':(np.zeros_like(vectors),None,0,0.0,0.0),
      'explicit_fp32':(vectors,None,0,0.0,0.0),
      'explicit_int8':(int8_decoded,(q,scales),0,0.0,0.0),
      'native_pca_r4':(pca_mean+pca_codes@pca_basis.T,(pca_mean,pca_basis,pca_codes),0,pca_fit_s,0.0),
      'mirror_learned_r4':(learned_mean+learned_codes@learned_basis.T,(learned_mean,learned_basis,learned_codes),2000,learned_fit_s,learned_assign_s)
    }
    rows=[]
    for name,(decoded,parts,updates,fit_s,assign_s) in methods.items():
        pay=payload(out/(name+'.npz'),name,vectors,parts)
        t=time.perf_counter();metrics=h.evaluate(model,tok,torch,vectors,decoded,manifest);infer_s=time.perf_counter()-t
        vm=vector_metrics(vectors,decoded) if name not in ('none','explicit_fp32') else {}
        row={'seed':seed,'method':name,'rank':RANK if 'r4' in name else 0,'payload_bytes':pay,
             'common_base_bytes':common_bytes,'total_deployment_bytes':common_bytes+pay,
             'support_examples':128,'support_forward_calls':support['forward_calls'],
             'support_input_tokens':support['input_tokens'],'optimizer_updates':updates,
             'fit_seconds':fit_s,'code_assignment_seconds':assign_s,'extraction_seconds':extraction_s,
             'inference_seconds':infer_s,'active_compute_proxy':(2000*12*512*4 if updates else 0)+(4*512*4 if 'r4' in name else 0)+(512*metrics['candidate_sequences'] if name != 'none' else 0),
             'metrics':metrics,'vector_metrics':vm}
        rows.append(row)
    report={'experiment_id':'MA-520','seed':seed,'model':'EleutherAI/pythia-70m-deduped','revision':REVISION,
            'model_sha256':MODEL_HASH,'dimension':512,'common_base_bytes':common_bytes,
            'task_ids':list(range(16)),'decoder_fit_task_ids':FIT_TASKS.tolist(),'heldout_task_ids':HELD_TASKS.tolist(),
            'extraction_seconds':extraction_s,'rows':rows}
    (out/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'seed':seed,'extraction_seconds':extraction_s,'learned_fit_seconds':learned_fit_s,
      'rows':[{'method':r['method'],'bytes':r['payload_bytes'],'accuracy':r['metrics']['heldout_accuracy'],
               'gold_logprob':r['metrics']['mean_gold_candidate_logprob'],**r['vector_metrics']} for r in rows]},indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--model-dir',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();run(a.seed,a.model_dir,a.out)
if __name__=='__main__':main()
