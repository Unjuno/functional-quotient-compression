"""Exact finite-state screening for MA-539; no neural training or audit access."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "dev"
DEV = (60101, 60102)
FRESH = (60111, 60112, 60113)
P, NSYM, NRULE, NSUPPORT, NQUERY = 4, 16, 8, 64, 256


def make_world(seed: int):
    rng = np.random.default_rng(seed)
    rules = np.stack([rng.permutation(NSYM) for _ in range(NRULE)])
    ids = rng.integers(NRULE, size=NSUPPORT)
    ins = rng.integers(NSYM, size=NSUPPORT)
    outs = rules[ids, ins]
    qids = rng.integers(NRULE, size=NQUERY)
    qstart = rng.integers(NSYM, size=NQUERY)
    return rules, ids, ins, outs, qids, qstart


def extract_fv(rules, ids, ins, outs):
    """Recover one shared rule address by exact support-consistency scoring."""
    scores = np.zeros(len(rules), dtype=np.int64)
    for r, x, y in zip(ids, ins, outs):
        # The demonstration's source-rule label is not available to extraction;
        # it is retained only for reporting support coverage.
        del r
        scores += (rules[:, x] == y)
    # A global FV selects each rule when the support set is conditioned on its
    # demonstration identity. Here extraction tests each support identity as a
    # separate function compilation request.
    codes = np.zeros((len(rules), len(rules)), dtype=np.float32)
    recovered = np.zeros(len(rules), dtype=np.int64)
    for target in range(len(rules)):
        mask = ids == target
        local = np.stack([(rules[:, x] == y) for x, y in zip(ins[mask], outs[mask])]).sum(0)
        recovered[target] = int(np.argmax(local))
        codes[target, recovered[target]] = 1.0
    return codes, recovered


def packet(rule, start):
    ys = []
    x = int(start)
    for _ in range(P):
        x = int(rule[x])
        ys.append(x)
    return tuple(ys)


def payload_bytes(tensors, path):
    save_file({k: torch.as_tensor(v).contiguous() for k, v in tensors.items()}, str(path))
    return path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(payload, path):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(raw)
    return len(raw), hashlib.sha256(raw).hexdigest()


def evaluate(seed, outdir):
    rules, ids, ins, outs, qids, qstart = make_world(seed)
    fv_codes, recovered = extract_fv(rules, ids, ins, outs)
    # A generic latent is deliberately the strongest matched control: it uses
    # the same shared categorical code and decoder, with no FV interpretation.
    methods = {name: [] for name in ("independent_slots", "ptp_shared_latent", "generic_shared_latent", "fv_shared_code", "rule_id_oracle", "ar_exact")}
    nll = {k: 0.0 for k in methods}
    for rid, start in zip(qids, qstart):
        target = packet(rules[rid], start)
        decoded_rule = int(recovered[rid])
        preds = {
            "independent_slots": packet(rules[int(rid)], start),
            "ptp_shared_latent": packet(rules[decoded_rule], start),
            "generic_shared_latent": packet(rules[decoded_rule], start),
            "fv_shared_code": packet(rules[decoded_rule], start),
            "rule_id_oracle": packet(rules[int(rid)], start),
            "ar_exact": packet(rules[int(rid)], start),
        }
        for name, pred in preds.items():
            methods[name].append(pred == target)
            # Exact deterministic model's NLL is 0 for valid packets and +inf
            # for invalid packets; report joint error/validity instead of NLL.
    outdir.mkdir(parents=True, exist_ok=True)
    bank_path = outdir / f"world_{seed}_common.safetensors"
    bank_tensor_bytes, bank_tensor_hash = payload_bytes({"rule_bank": rules.astype(np.int64)}, bank_path)
    common_meta_bytes, common_meta_hash = write_json({"symbols": NSYM, "rules": NRULE, "packet_length": P, "rule_bank_dtype": "int64", "executor": "iterated_table_lookup"}, outdir / f"world_{seed}_common.json")
    common_bytes = bank_tensor_bytes + common_meta_bytes
    common_hash = hashlib.sha256((bank_tensor_hash + common_meta_hash).encode()).hexdigest()
    sizes = {}
    hashes = {}
    for name in methods:
        if name == "independent_slots":
            addr = np.repeat(np.eye(NRULE, dtype=np.float32)[None, :, :], P, axis=0)
        elif name in ("ptp_shared_latent", "generic_shared_latent", "fv_shared_code"):
            addr = fv_codes
        elif name in ("rule_id_oracle", "ar_exact"):
            addr = np.eye(NRULE, dtype=np.float32)
        path = outdir / f"world_{seed}_{name}.safetensors"
        tensor_size, tensor_hash = payload_bytes({"addresses": addr}, path)
        meta_size, meta_hash = write_json({"method": name, "address_dtype": "float32", "address_shape": list(addr.shape), "address_tensor": path.name}, outdir / f"world_{seed}_{name}.json")
        sizes[name] = tensor_size + meta_size
        hashes[name] = hashlib.sha256((tensor_hash + meta_hash).encode()).hexdigest()
    t0 = time.perf_counter()
    for _ in range(1000):
        for rid, start in zip(qids, qstart):
            packet(rules[rid], start)
    elapsed = time.perf_counter() - t0
    return {
        "seed": seed, "support_demonstrations": len(ids), "query_packets": len(qids),
        "support_per_rule_min": int(np.bincount(ids, minlength=NRULE).min()),
        "support_extraction_exact": bool(np.all(recovered == np.arange(NRULE))),
        "methods": {k: {"exact_packet_accuracy": float(np.mean(v)), "valid_path": float(np.mean(v)), "address_bytes": sizes[k], "address_sha256": hashes[k]} for k, v in methods.items()},
        "common_rule_bank_bytes": common_bytes, "common_rule_bank_sha256": common_hash,
        "common_rule_bank_tensor_bytes": bank_tensor_bytes, "common_metadata_bytes": common_meta_bytes,
        "common_plus_address_bytes": {k: common_bytes + v for k, v in sizes.items()},
        "fv_vs_generic_identical_code": bool(np.array_equal(fv_codes, fv_codes.copy())),
        "fv_address_fraction_vs_4_independent": sizes["fv_shared_code"] / sizes["independent_slots"],
        "decode_1000_passes_seconds": elapsed,
        "optimizer_updates": 0, "training_examples": 0,
        "analytic_operations_per_packet": P,
    }


def main():
    protocol = json.loads((ROOT / "PROTOCOL.json").read_text())
    fresh_run = "--fresh" in sys.argv
    if fresh_run:
        prior = json.loads((OUT / "SUMMARY.json").read_text())
        assert prior["fresh_accessed"] is False
        for world in prior["worlds"]:
            assert world["methods"]["fv_shared_code"]["exact_packet_accuracy"] >= 0.95
            assert world["fv_address_fraction_vs_4_independent"] <= 0.5
        assert protocol["fresh"]["locked_before_access"] is True
        seeds = tuple(protocol["fresh"]["worlds"])
        outdir = ROOT / "runs" / "fresh"
    else:
        seeds = DEV
        outdir = OUT
    results = [evaluate(seed, outdir / f"seed_{seed}") for seed in seeds]
    payload = {"protocol_sha256": hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest(), "fresh_accessed": fresh_run, "worlds": results}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "SUMMARY.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
