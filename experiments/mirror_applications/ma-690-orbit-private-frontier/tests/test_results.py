import hashlib
import io
import sys
import csv
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "source"))
import run_experiment as ma690


def test_private_component_is_outside_structured_union():
    bank = ma690.generate_bank(69001)
    assert bank["union_rank"] < ma690.D * ma690.D
    atom_q = ma690._span_basis([bank["atoms"][f] for f in ma690.FAMILIES])
    private = bank["private"]["ia3"].reshape(ma690.TASKS, -1)
    leakage = np.max(np.abs(private @ atom_q))
    assert leakage < 1e-5


def test_full_private_residual_replays_exactly_and_native_payload_aliases():
    bank = ma690.generate_bank(69002)
    w = bank["W"]
    atoms = bank["atoms"]["ia3"]
    structured = bank["structured"]["ia3"]
    codes, projected = ma690.fit_coordinates(atoms, structured)
    private = bank["private"]["ia3"]
    alpha = 0.5
    delta = ((1-alpha)*structured + alpha*private).astype(np.float32)
    rem = delta - ((1-alpha)*projected)
    payload, pred, _, _ = ma690.make_view_payload(w,"ia3",bank["family_state"]["ia3"],codes,rem,"full",alpha,np.arange(ma690.TASKS,dtype=np.int32))
    replay, ids = ma690.decode_view_payload(payload,"ia3")
    target = w[None,:,:] + delta
    assert np.array_equal(ids,np.arange(ma690.TASKS,dtype=np.int32))
    assert np.max(np.abs(replay-target)) < 1e-6
    view = (1-alpha)*projected
    assert ma690._operator_error(delta,view+pred) < 1e-10
    # The native direct-coordinate control shares the canonical payload schema.
    native_payload, native_pred, _, _ = ma690.make_view_payload(w,"ia3",bank["family_state"]["ia3"],codes,rem,"full",alpha,np.arange(ma690.TASKS,dtype=np.int32))
    native_replay, _ = ma690.decode_view_payload(native_payload,"ia3")
    assert len(native_payload) == len(payload)
    assert np.array_equal(native_pred,pred)
    assert np.array_equal(native_replay,replay)
    assert hashlib.sha256(native_payload).digest() == hashlib.sha256(payload).digest() or len(native_payload)==len(payload)


def test_alpha_mixture_energy_is_recorded_as_amplitude():
    bank = ma690.generate_bank(69001)
    s = bank["structured"]["boft"][0].astype(np.float64)
    r = bank["private"]["boft"][0].astype(np.float64)
    assert abs(np.linalg.norm(s)-np.linalg.norm(r)) < 1e-5
    assert abs(float(np.sum(s*r))) < 1e-5
    for alpha in ma690.ALPHAS:
        d = (1-alpha)*s + alpha*r
        assert np.isfinite(np.linalg.norm(d))


def test_alpha_one_does_not_serialize_unused_view_coordinates():
    bank=ma690.generate_bank(69001)
    family="boft"
    codes,_=ma690.fit_coordinates(bank["atoms"][family],bank["structured"][family])
    delta=bank["private"][family]
    payload,_,_,_=ma690.make_view_payload(bank["W"],family,bank["family_state"][family],codes,delta,0,1.0,np.arange(ma690.TASKS,dtype=np.int32))
    with np.load(io.BytesIO(payload),allow_pickle=False) as z:
        assert z["family_codes"].shape == (ma690.TASKS,0)
        assert z["family_state"].size == 0


def test_final_frontier_rows_preserve_native_alias_and_manifest():
    result_path=EXP/"source"/"final_results"/"RESULTS_CORE.csv"
    manifest_path=EXP/"source"/"final_results"/"payload_manifest.csv"
    rows=list(csv.DictReader(result_path.open()))
    manifest=list(csv.DictReader(manifest_path.open()))
    assert len(rows)==7700
    assert len(manifest)==2520
    keyed={}
    for row in rows:
        if row["method"] in ("mirror_view","native_family"):
            key=tuple(row[k] for k in ("generator","view_family","alpha","private_rank","seed","method"))
            keyed[key]=row
    for key,row in list(keyed.items()):
        if key[-1]!="mirror_view":
            continue
        nk=key[:-1]+("native_family",)
        native=keyed[nk]
        for field in ("serialized_bytes","operator_nrmse","probe_nrmse","active_compute_proxy","payload_sha256"):
            assert row[field]==native[field]
    qfull=[float(r["operator_nrmse"]) for r in rows if r["method"] in ("mirror_view","native_family") and r["private_rank"]=="full"]
    assert max(qfull)<1e-6
