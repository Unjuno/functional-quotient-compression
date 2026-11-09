#!/usr/bin/env python3
"""MA-690: deterministic cross-family local operator-bank frontier."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
D = 16
TASKS = 8
SEEDS = (69001, 69002)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
RANKS: tuple[int | str, ...] = (0, 1, 2, 4, 8, "full")
FAMILIES = ("ia3", "batchensemble", "vera", "lowrank", "givens", "boft")
GENERATORS = FAMILIES + ("neutral",)
FAMILY_IDS = {name: i for i, name in enumerate(FAMILIES)}


def serialize(**arrays: np.ndarray) -> bytes:
    buff = io.BytesIO()
    np.savez_compressed(buff, **arrays)
    return buff.getvalue()


def decode_view_payload(payload: bytes, family: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as z:
        w = z["base_W"]
        alpha = float(z["alpha"][0])
        ids = z["task_ids"]
        codes = z["family_codes"]
        state = z["family_state"]
        if alpha == 1.0:
            view = np.zeros((len(ids), D, D), np.float32)
        else:
            atoms, _ = _atoms_from_state(family, w, state)
            view = (1.0-alpha) * np.einsum("nk,kij->nij", codes, atoms, optimize=True)
        if "residual_dense" in z.files:
            residual = z["residual_dense"]
        else:
            left, singular, right = z["residual_left"], z["residual_singular"], z["residual_right"]
            residual = np.zeros((len(ids), D, D), np.float32)
            for i in range(len(ids)):
                if singular.shape[1]:
                    residual[i] = (left[i] * singular[i][None, :]) @ right[i]
        return w[None, :, :] + view + residual, ids


def _schedule_givens(d: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(0, d, 2)]


def _schedule_boft(d: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for stage in range(int(np.log2(d))):
        step = 1 << stage
        pairs.extend((i, i ^ step) for i in range(d) if not (i & step))
    return pairs


def family_atoms(name: str, w: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return analytic local operator atoms [K,D,D] and paid random state."""
    d = w.shape[0]
    atoms: list[np.ndarray] = []
    state = np.zeros((0,), dtype=np.float32)
    if name == "ia3":
        for j in range(d):
            a = np.zeros_like(w)
            a[:, j] = w[:, j]
            atoms.append(a)
    elif name == "batchensemble":
        # First derivative at r=s=1; the O(r*s) cross term is excluded by scope.
        for i in range(d):
            a = np.zeros_like(w)
            a[i, :] = w[i, :]
            atoms.append(a)
        for j in range(d):
            a = np.zeros_like(w)
            a[:, j] = w[:, j]
            atoms.append(a)
    elif name == "vera":
        rng = np.random.default_rng(seed)
        r = 4
        b = (rng.standard_normal((d, r)) / np.sqrt(d)).astype(np.float32)
        a = (rng.standard_normal((r, d)) / np.sqrt(r)).astype(np.float32)
        state = np.concatenate((b.ravel(), a.ravel())).astype(np.float32)
        for k in range(r):
            atoms.append(np.outer(b[:, k], a[k, :]).astype(np.float32))
    elif name == "lowrank":
        rng = np.random.default_rng(seed)
        r = 2
        u = (rng.standard_normal((d, r)) / np.sqrt(d)).astype(np.float32)
        v = (rng.standard_normal((r, d)) / np.sqrt(r)).astype(np.float32)
        state = np.concatenate((u.ravel(), v.ravel())).astype(np.float32)
        for i in range(r):
            for j in range(r):
                atoms.append(np.outer(u[:, i], v[j, :]).astype(np.float32))
    elif name in ("givens", "boft"):
        pairs = _schedule_givens(d) if name == "givens" else _schedule_boft(d)
        for i, j in pairs:
            a = np.zeros_like(w)
            a[i, :] = -w[j, :]
            a[j, :] = w[i, :]
            atoms.append(a)
    else:
        raise ValueError(name)
    return np.stack(atoms).astype(np.float32), state


def _span_basis(atom_banks: list[np.ndarray]) -> np.ndarray:
    mat = np.concatenate([a.reshape(a.shape[0], -1).T for a in atom_banks], axis=1).astype(np.float64)
    u, s, _ = np.linalg.svd(mat, full_matrices=False)
    if not len(s):
        return np.zeros((D * D, 0), dtype=np.float64)
    tol = max(mat.shape) * np.finfo(float).eps * s[0] * 100
    return u[:, s > tol]


def _orthogonal_random(rng: np.random.Generator, q_union: np.ndarray, additionally: list[np.ndarray] | None = None) -> np.ndarray:
    for _ in range(100):
        v = rng.standard_normal(D * D)
        if q_union.size:
            v -= q_union @ (q_union.T @ v)
        if additionally:
            for a in additionally:
                v -= a * float(a @ v)
        n = np.linalg.norm(v)
        if n > 1e-10:
            return v / n
    raise RuntimeError("orthogonal complement too small for requested private directions")


def _normalize(mat: np.ndarray, target_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(mat))
    if n < 1e-12:
        raise RuntimeError("degenerate family update")
    return (mat * (target_norm / n)).astype(np.float32)


def generate_bank(seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((D, D)) / np.sqrt(D)).astype(np.float32)
    atom_banks, family_state = {}, {}
    for f in FAMILIES:
        atom_banks[f], family_state[f] = family_atoms(f, w, seed + 1000 + FAMILY_IDS[f])
    q_union = _span_basis([atom_banks[f] for f in FAMILIES])
    strength = 0.1 * float(np.linalg.norm(w))
    structured: dict[str, np.ndarray] = {}
    private: dict[str, np.ndarray] = {}
    for g_idx, g in enumerate(GENERATORS):
        sg = np.zeros((TASKS, D, D), dtype=np.float32)
        rg = np.zeros_like(sg)
        for task in range(TASKS):
            if g == "neutral":
                svec = _orthogonal_random(rng, q_union)
                smat = svec.reshape(D, D)
            else:
                ag = atom_banks[g]
                coeff = rng.standard_normal(ag.shape[0]).astype(np.float32)
                smat = np.einsum("k,kij->ij", coeff, ag, optimize=True)
            smat = _normalize(smat, strength)
            svec = smat.astype(np.float64).ravel()
            rvec = _orthogonal_random(rng, q_union, [svec / np.linalg.norm(svec)])
            sg[task] = smat
            rg[task] = (rvec.reshape(D, D) * strength).astype(np.float32)
        structured[g], private[g] = sg, rg
    return {"W": w, "atoms": atom_banks, "family_state": family_state, "structured": structured, "private": private, "union_rank": int(q_union.shape[1])}


def fit_coordinates(atoms: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = atoms.reshape(atoms.shape[0], -1).T.astype(np.float64)
    codes, *_ = np.linalg.lstsq(flat, targets.reshape(targets.shape[0], -1).T.astype(np.float64), rcond=1e-10)
    pred = (flat @ codes).T.reshape(targets.shape).astype(np.float32)
    return codes.T.astype(np.float32), pred


def _svd_residual(residuals: np.ndarray, rank: int | str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n = residuals.shape[0]
    if rank == "full":
        dense = residuals.astype(np.float32)
        return dense, {"residual_dense": dense}
    q = int(rank)
    left = np.zeros((n, D, q), np.float32)
    singular = np.zeros((n, q), np.float32)
    right = np.zeros((n, q, D), np.float32)
    recon = np.zeros_like(residuals, dtype=np.float32)
    if q:
        for i in range(n):
            u, s, vt = np.linalg.svd(residuals[i].astype(np.float64), full_matrices=False)
            left[i] = u[:, :q].astype(np.float32)
            singular[i] = s[:q].astype(np.float32)
            right[i] = vt[:q].astype(np.float32)
            recon[i] = (left[i] * singular[i][None, :]) @ right[i]
    return recon, {"residual_left": left, "residual_singular": singular, "residual_right": right}


def _operator_error(target_delta: np.ndarray, pred_delta: np.ndarray) -> float:
    den = float(np.sum(target_delta.astype(np.float64) ** 2))
    return float(np.sum((target_delta.astype(np.float64) - pred_delta.astype(np.float64)) ** 2) / max(den, 1e-20))


def _probe_error(target_delta: np.ndarray, pred_delta: np.ndarray, probes: np.ndarray) -> float:
    err = (target_delta - pred_delta) @ probes.T
    ref = target_delta @ probes.T
    return float(np.sum(err.astype(np.float64) ** 2) / max(float(np.sum(ref.astype(np.float64) ** 2)), 1e-20))


def _base_payload(w: np.ndarray, task_ids: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    return {"base_W": w.astype(np.float32), "task_ids": task_ids.astype(np.int32), "alpha": np.array([alpha], np.float32), "schema_version": np.array([1], np.uint8)}


def make_view_payload(w: np.ndarray, family: str, state: np.ndarray, codes: np.ndarray, residuals: np.ndarray, rank: int | str, alpha: float, task_ids: np.ndarray) -> tuple[bytes, np.ndarray, int, int]:
    rec_res, res_fields = _svd_residual(residuals, rank)
    atoms, _ = family_atoms(family, w, 0) if family not in ("vera", "lowrank") else (None, None)
    # Random shared basis tensors are paid unless alpha=1 makes the view unused.
    paid_state = state.astype(np.float32) if alpha < 1.0 else np.zeros((0,), np.float32)
    extra = {"family_state": paid_state}
    if atoms is None:
        atoms, _ = _atoms_from_state(family, w, state)
    code_fields = np.zeros((TASKS, 0), np.float32) if alpha == 1.0 else codes.astype(np.float32)
    fields = _base_payload(w, task_ids, alpha)
    fields.update({"family_id": np.array([FAMILY_IDS[family]], np.uint8), "family_codes": code_fields, **extra, **res_fields})
    payload = serialize(**fields)
    view = ((1.0-alpha) * np.einsum("nk,kij->nij", code_fields, atoms, optimize=True)
            if alpha < 1.0 else np.zeros((TASKS,D,D),np.float32))
    return payload, rec_res.astype(np.float32), int(atoms.shape[0] * D * D if alpha < 1 else 0), int(rank if isinstance(rank, int) else 0) * D * D


def _atoms_from_state(family: str, w: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = w.shape[0]
    if family == "vera":
        r = 4
        b = state[:d*r].reshape(d, r)
        a = state[d*r:].reshape(r, d)
        return np.stack([np.outer(b[:, k], a[k]) for k in range(r)]).astype(np.float32), state
    if family == "lowrank":
        r = 2
        u = state[:d*r].reshape(d, r)
        v = state[d*r:].reshape(r, d)
        return np.stack([np.outer(u[:, i], v[j]) for i in range(r) for j in range(r)]).astype(np.float32), state
    return family_atoms(family, w, 0)


def make_residual_only_payload(w: np.ndarray, task_ids: np.ndarray, alpha: float, delta: np.ndarray, rank: int | str) -> tuple[bytes, np.ndarray, int]:
    recon, fields_r = _svd_residual(delta, rank)
    fields = _base_payload(w, task_ids, alpha)
    fields.update(fields_r)
    return serialize(**fields), recon, int(rank if isinstance(rank, int) else 0) * D * D


def make_independent_payload(matrices: np.ndarray, task_ids: np.ndarray) -> bytes:
    return serialize(task_matrices=matrices.astype(np.float32), task_ids=task_ids.astype(np.int32), schema_version=np.array([1], np.uint8))


def _runtime_bank(w: np.ndarray, atoms: np.ndarray, codes: np.ndarray, rem: np.ndarray, probes: np.ndarray) -> float:
    start = time.perf_counter()
    for _ in range(10):
        mats = w[None, :, :] + np.einsum("nk,kij->nij", codes, atoms, optimize=True) + rem
        _ = np.einsum("nij,pj->nip", mats, probes, optimize=True)
    return (time.perf_counter() - start) / 10.0


def run(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    all_union_ranks: dict[str, int] = {}
    rng_probe = np.random.default_rng(69000)
    probes = (rng_probe.standard_normal((64, D)) / np.sqrt(D)).astype(np.float32)
    task_ids = np.arange(TASKS, dtype=np.int32)

    for seed in SEEDS:
        bank = generate_bank(seed)
        w = bank["W"]
        atoms_bank = bank["atoms"]
        state_bank = bank["family_state"]
        structured = bank["structured"]
        private = bank["private"]
        all_union_ranks[str(seed)] = bank["union_rank"]
        for gen in GENERATORS:
            sg, rg = structured[gen], private[gen]
            for alpha in ALPHAS:
                delta = ((1.0-alpha)*sg + alpha*rg).astype(np.float32)
                targets = w[None, :, :] + delta
                independent_payload = make_independent_payload(targets, task_ids)
                independent_bytes = len(independent_payload)
                full_probe = np.einsum("nij,pj->nip", targets, probes, optimize=True)
                baseline_apply_ops = D * D
                base_payload = serialize(**_base_payload(w, task_ids, alpha))
                tstart=time.perf_counter()
                for _ in range(10):
                    _=np.einsum("nij,pj->nip",targets,probes,optimize=True)
                independent_runtime=(time.perf_counter()-tstart)/10
                result_rows.append({"generator":gen,"view_family":"none","alpha":alpha,"private_rank":"none","seed":seed,"method":"independent_full","serialized_bytes":independent_bytes,"bytes_per_task":independent_bytes/TASKS,"operator_nrmse":0.0,"probe_nrmse":0.0,"view_decode_ops_proxy":0,"private_decode_ops_proxy":0,"apply_ops_proxy":baseline_apply_ops,"active_compute_proxy":baseline_apply_ops,"fit_wall_s":0.0,"serialization_wall_s":0.0,"serve_wall_s":independent_runtime,"residual_only_bytes":"","residual_only_error":"","payload_sha256":hashlib.sha256(independent_payload).hexdigest(),"notes":"exact full target matrix bank"})
                result_rows.append({"generator":gen,"view_family":"none","alpha":alpha,"private_rank":"none","seed":seed,"method":"backbone_only","serialized_bytes":len(base_payload),"bytes_per_task":len(base_payload)/TASKS,"operator_nrmse":_operator_error(delta,np.zeros_like(delta)),"probe_nrmse":_probe_error(delta,np.zeros_like(delta),probes),"view_decode_ops_proxy":0,"private_decode_ops_proxy":0,"apply_ops_proxy":baseline_apply_ops,"active_compute_proxy":baseline_apply_ops,"fit_wall_s":0.0,"serialization_wall_s":0.0,"serve_wall_s":0.0,"residual_only_bytes":"","residual_only_error":"","payload_sha256":hashlib.sha256(base_payload).hexdigest(),"notes":"shared W only"})

                for f in FAMILIES:
                    atoms = atoms_bank[f]
                    state = state_bank[f]
                    t0 = time.perf_counter()
                    codes, projected_s = fit_coordinates(atoms, sg)
                    fit_s = time.perf_counter() - t0
                    # Fit the view to the structured component. The private component stays in the residual.
                    shared_delta = ((1.0-alpha)*projected_s).astype(np.float32)
                    remainder = (delta - shared_delta).astype(np.float32)
                    native_delta = shared_delta.copy()
                    native_res = remainder

                    candidate_cache: dict[str, tuple[bytes,np.ndarray,int,int,float]] = {}
                    residual_cache: dict[str, tuple[bytes,np.ndarray,int]] = {}
                    for q in RANKS:
                        serial_start = time.perf_counter()
                        payload, rec, view_ops, priv_ops = make_view_payload(w,f,state,codes,remainder,q,alpha,task_ids)
                        serial_s = time.perf_counter()-serial_start
                        candidate_cache[str(q)] = (payload,rec,view_ops,priv_ops,serial_s)
                        payload_hash = hashlib.sha256(payload).hexdigest()
                        manifest_rows.append({"seed":seed,"generator":gen,"family":f,"alpha":alpha,"rank":q,"kind":"mirror_and_native_identical","bytes":len(payload),"sha256":payload_hash})
                        # Native family coordinates use the identical representation and payload schema.
                        for method in ("mirror_view","native_family"):
                            prediction = (shared_delta + rec).astype(np.float32)
                            err = _operator_error(delta,prediction)
                            p_err = _probe_error(delta,prediction,probes)
                            decoded_full = prediction
                            runtime_atoms = atoms if alpha < 1.0 else np.zeros((0,D,D),np.float32)
                            runtime_codes = codes*(1-alpha) if alpha < 1.0 else np.zeros((TASKS,0),np.float32)
                            runtime = _runtime_bank(w, runtime_atoms, runtime_codes, rec, probes)
                            apply_ops = baseline_apply_ops
                            active = apply_ops + int(np.ceil((view_ops+priv_ops)/128))
                            result_rows.append({"generator":gen,"view_family":f,"alpha":alpha,"private_rank":q,"seed":seed,"method":method,"serialized_bytes":len(payload),"bytes_per_task":len(payload)/TASKS,"operator_nrmse":float(np.sqrt(max(err,0))),"probe_nrmse":float(np.sqrt(max(p_err,0))),"view_decode_ops_proxy":view_ops,"private_decode_ops_proxy":priv_ops,"apply_ops_proxy":apply_ops,"active_compute_proxy":active,"fit_wall_s":fit_s,"serialization_wall_s":serial_s,"serve_wall_s":runtime,"residual_only_bytes":"","residual_only_error":"","payload_sha256":payload_hash,"notes":"native and Mirror use identical direct family coordinate; oracle least-squares and SVD"})

                    # Residual-only SVD variants, compared at each candidate's actual payload budget.
                    for q in RANKS:
                        p, rec, priv_ops = make_residual_only_payload(w,task_ids,alpha,delta,q)
                        residual_cache[str(q)] = (p,rec,priv_ops)
                    for q in RANKS:
                        payload, rec, view_ops, priv_ops, serial_s = candidate_cache[str(q)]
                        feasible=[]
                        for rq in RANKS:
                            rp, rr, rops = residual_cache[str(rq)]
                            if len(rp) <= len(payload):
                                feasible.append(( _operator_error(delta,rr), len(rp), rq, rp, rr, rops))
                        if feasible:
                            best=min(feasible,key=lambda z:(z[0],z[1]))
                            res_err,res_bytes,res_rank,res_payload,res_pred,res_ops=best
                            res_probe=_probe_error(delta,res_pred,probes)
                            res_runtime=_runtime_bank(w,np.zeros((0,D,D),np.float32),np.zeros((TASKS,0),np.float32),res_pred,probes)
                            result_rows.append({"generator":gen,"view_family":f,"alpha":alpha,"private_rank":q,"seed":seed,"method":"residual_only_matched_budget","serialized_bytes":res_bytes,"bytes_per_task":res_bytes/TASKS,"operator_nrmse":float(np.sqrt(max(res_err,0))),"probe_nrmse":float(np.sqrt(max(res_probe,0))),"view_decode_ops_proxy":0,"private_decode_ops_proxy":res_ops,"apply_ops_proxy":baseline_apply_ops,"active_compute_proxy":baseline_apply_ops+int(np.ceil(res_ops/128)),"fit_wall_s":0.0,"serialization_wall_s":0.0,"serve_wall_s":res_runtime,"residual_only_bytes":res_bytes,"residual_only_error":float(np.sqrt(max(res_err,0))),"payload_sha256":hashlib.sha256(res_payload).hexdigest(),"notes":f"rank={res_rank}; largest quality point fitting candidate payload bytes"})

    out_csv=out_dir/"RESULTS_CORE.csv"
    fields=["generator","view_family","alpha","private_rank","seed","method","serialized_bytes","bytes_per_task","operator_nrmse","probe_nrmse","view_decode_ops_proxy","private_decode_ops_proxy","apply_ops_proxy","active_compute_proxy","fit_wall_s","serialization_wall_s","serve_wall_s","residual_only_bytes","residual_only_error","payload_sha256","notes"]
    with out_csv.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(result_rows)
    with (out_dir/"payload_manifest.csv").open('w',newline='',encoding='utf-8') as f:
        mf=list(manifest_rows[0].keys()) if manifest_rows else []
        writer=csv.DictWriter(f,fieldnames=mf);writer.writeheader();writer.writerows(manifest_rows)
    summary={"experiment_id":"MA-690","protocol_sha256":hashlib.sha256((EXP/"PROTOCOL.json").read_bytes()).hexdigest(),"development_seeds":list(SEEDS),"fresh_opened":False,"rows":len(result_rows),"mirror_native_exact_payload_rows":len(manifest_rows),"structured_union_ranks":all_union_ranks,"families":list(FAMILIES),"generators":list(GENERATORS),"alphas":list(ALPHAS),"ranks":list(RANKS)}
    (out_dir/"run_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding='utf-8')


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=EXP/'source'/'results')
    args=parser.parse_args()
    run(args.out)


if __name__=='__main__':
    main()
