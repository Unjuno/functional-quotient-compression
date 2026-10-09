#!/usr/bin/env python3
"""Deterministic synthetic conditional activation-steering screen for MA-510."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

D, C, B = 64, 16, 8
CRANKS, BRANKS = (2, 4, 8, 16), (2, 4, 8)

def orth(rng, n, k):
    q, _ = np.linalg.qr(rng.normal(size=(n, k)))
    return q[:, :k].astype(np.float32)

def make_world(seed):
    rng = np.random.default_rng(seed)
    qx = orth(rng, D, 4); qv = orth(rng, D, 4)
    latent = rng.normal(size=(C, 4)).astype(np.float32)
    latent /= np.maximum(np.linalg.norm(latent, axis=1, keepdims=True), 1e-8)
    keys = (latent @ qx.T + .015 * orth(rng, D, C).T).astype(np.float32)
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    coeff = rng.normal(size=(B, 4)).astype(np.float32)
    effects = (coeff @ qv.T).astype(np.float32)
    effects /= np.maximum(np.linalg.norm(effects, axis=1, keepdims=True), 1e-8)
    mapping = np.arange(C, dtype=np.uint16) % B
    def positives(n):
        ids = np.repeat(np.arange(C), n)
        x = keys[ids] + .22 * rng.normal(size=(len(ids), D)).astype(np.float32)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
        return x, ids
    def negatives(n):
        # Closest non-trigger contexts: midpoints of two condition prototypes.
        ids = rng.integers(0, C, size=(n, 2))
        ids[:, 1] = (ids[:, 0] + rng.integers(1, C, size=n)) % C
        x = .5 * (keys[ids[:, 0]] + keys[ids[:, 1]])
        x += .22 * rng.normal(size=(n, D)).astype(np.float32)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
        return x
    sx, sy = positives(128)
    cp, cy = positives(64); cn = negatives(2048)
    ep, ey = positives(256); en = negatives(8192)
    return dict(keys=keys, effects=effects, mapping=mapping, support_x=sx, support_y=sy,
                cal_pos=cp, cal_y=cy, cal_neg=cn, eval_pos=ep, eval_y=ey, eval_neg=en)

def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

def svd_basis(rows, rank):
    # Rows are (objects, D); no centering so this remains a linear intervention basis.
    _, _, vt = np.linalg.svd(rows.astype(np.float64), full_matrices=False)
    return vt[:rank].T.astype(np.float32)

def fit_gate(x, labels, neg, mode, rank_c, rank_b, world):
    # All representations derive only from the frozen support prototypes / intervention contrasts.
    prot = np.stack([x[labels == c].mean(axis=0) for c in range(C)]).astype(np.float32)
    prot = normalize(prot)
    beh = world['effects'].copy()
    if mode in ('mirror', 'native_pca'):
        bc = svd_basis(prot, rank_c)
        zc = prot @ bc
        decoded = normalize(zc @ bc.T)
        bb = svd_basis(beh, rank_b)
        zb = beh @ bb
        act = (zb @ bb.T).astype(np.float32)
    elif mode in ('cast', 'independent_pair'):
        bc, zc, decoded, bb, act = None, None, prot, None, beh
    else: raise ValueError(mode)
    # Both factorized Mirror and native PCA use identical transforms, codes and gate function.
    def scores(a): return normalize(a) @ decoded.T
    cal_scores = scores(world['cal_pos'])
    cal_pos_best = cal_scores.max(axis=1)
    cal_neg_best = scores(neg).max(axis=1)
    # Lowest threshold achieving <=1% calibration FPR; deterministic empirical quantile.
    threshold = float(np.quantile(cal_neg_best, .99, method='higher'))
    return dict(prototypes=decoded.astype(np.float32), actions=act.astype(np.float32),
                basis_c=bc, codes_c=zc, basis_b=bb, codes_b=zb if mode in ('mirror','native_pca') else None,
                threshold=threshold, score_fn=scores, calibration_pos=cal_pos_best)

def payload_bytes(path, arrays):
    np.savez(path, **{k:v for k,v in arrays.items() if v is not None})
    return Path(path).stat().st_size

def payload_for(path, mode, fitted, world, rank_c, rank_b):
    if mode in ('mirror','native_pca'):
        arr=dict(condition_basis=fitted['basis_c'], condition_codes=fitted['codes_c'],
                 behavior_basis=fitted['basis_b'], behavior_codes=fitted['codes_b'],
                 condition_behavior_address=world['mapping'], gate_threshold=np.array([fitted['threshold']],np.float32),
                 schema=np.array([510,rank_c,rank_b],np.int32))
    elif mode=='cast':
        arr=dict(condition_vectors=fitted['prototypes'],behavior_vectors=fitted['actions'],
                 condition_behavior_address=world['mapping'],gate_threshold=np.array([fitted['threshold']],np.float32),schema=np.array([510,0,0],np.int32))
    elif mode=='independent_pair':
        arr=dict(condition_vectors=fitted['prototypes'],pair_behavior_vectors=fitted['actions'][world['mapping']],
                 condition_behavior_address=world['mapping'],gate_threshold=np.array([fitted['threshold']],np.float32),schema=np.array([510,0,0],np.int32))
    return payload_bytes(path,arr)

def evaluate(f, world):
    pred_scores=f['score_fn'](world['eval_pos']); conf=pred_scores.max(axis=1); guess=pred_scores.argmax(axis=1)
    trigger=conf >= f['threshold']
    recall=float(trigger.mean())
    bacc=float(np.mean(world['mapping'][guess[trigger]]==world['mapping'][world['eval_y'][trigger]])) if trigger.any() else 0.
    neg_conf=f['score_fn'](world['eval_neg']).max(axis=1)
    fpr=float(np.mean(neg_conf >= f['threshold']))
    precision=float(trigger.sum()/(trigger.sum()+np.sum(neg_conf>=f['threshold'])))
    # Compare actual intervention outputs to target vector; misses and wrong routes receive zero.
    got=np.zeros((len(guess),D),np.float32)
    got[trigger]=f['actions'][world['mapping'][guess[trigger]]]
    target=f['actions'][world['mapping'][world['eval_y']]]
    rel=float(np.linalg.norm(got-target)/max(np.linalg.norm(target),1e-12))
    return dict(event_recall=recall,conditional_behavior_accuracy=bacc,false_trigger_rate=fpr,
                trigger_precision=precision,conditional_output_relative_rmse=rel,
                unique_conditional_outputs=int(len(np.unique(np.round(got[trigger],6),axis=0))) if trigger.any() else 0)

def one(seed, out):
    w=make_world(seed)
    # Estimate the condition prototypes using support activations rather than teacher keys.
    modes=[('cast',0,0),('independent_pair',0,0)] + [('mirror',rc,rb) for rc in CRANKS for rb in BRANKS] + [('native_pca',rc,rb) for rc in CRANKS for rb in BRANKS]
    rows=[]; payloads={}
    for mode,rc,rb in modes:
        fit_start=time.perf_counter()
        # Fit by support per condition; shared codes reconstruct inferred prototypes.
        fitted=fit_gate(w['support_x'],w['support_y'],w['cal_neg'],mode,rc,rb,w)
        key=f'{mode}_c{rc}_b{rb}'
        fit_seconds=time.perf_counter()-fit_start
        p=Path(out)/f'{key}.npz'; nbytes=payload_for(p,mode,fitted,w,rc,rb); payloads[key]=nbytes
        infer_start=time.perf_counter(); metrics=evaluate(fitted,w); infer_seconds=time.perf_counter()-infer_start
        if mode=='native_pca':
            mirror_key=f'mirror_c{rc}_b{rb}'
            alias=(nbytes==payloads[mirror_key] and metrics==next(r['metrics'] for r in rows if r['key']==mirror_key))
        else: alias=False
        d=rc if mode in ('mirror','native_pca') else D
        ops=int(D*rc + C*rc + D*rb + D) if mode in ('mirror','native_pca') else int(C*D + D)
        rows.append(dict(seed=seed,key=key,mode=mode,condition_rank=rc,behavior_rank=rb,bytes=nbytes,
                         ops_per_example=ops,fit_seconds=fit_seconds,inference_seconds=infer_seconds,metrics=metrics,native_alias=alias))
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); Path(a.out).mkdir(parents=True,exist_ok=True)
    t=time.perf_counter(); rows=one(a.seed,a.out)
    report=dict(experiment_id='MA-510',seed=a.seed,world_type='synthetic frozen activation mechanism screen',
                fit_and_eval_seconds=time.perf_counter()-t,rows=rows)
    Path(a.out,'metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'seed':a.seed,'rows':len(rows),'seconds':report['fit_and_eval_seconds'],'primary':[r for r in rows if r['key'] in ('cast_c0_b0','mirror_c4_b4','native_pca_c4_b4')]},indent=2))
if __name__=='__main__': main()
