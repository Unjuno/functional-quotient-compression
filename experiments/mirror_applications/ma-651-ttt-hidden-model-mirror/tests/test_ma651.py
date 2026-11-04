import json
from pathlib import Path

import numpy as np

from run_experiment import ROOT, choose_mode, make_world, online_full, online_lowrank, RANK


def test_grid_storage_and_router_selects_correct_subspaces():
    rows=json.loads((ROOT/"source"/"development.json").read_text());assert len(rows)==10
    for row in rows:
        p=ROOT/"source"/f"{row['seed']}_{row['method']}.npz"
        assert p.stat().st_size==row["serialized_state_bytes"]
    for seed in (65101,65102):
        base,dictionary,seqs=make_world(seed)
        codes=[choose_mode(base,dictionary,s["x"][:8],s["y"][:8])[0] for s in seqs]
        assert codes==[s["mode"] for s in seqs]


def test_native_ids_equal_mirror_state_and_outputs():
    for seed in (65101,65102):
        a=ROOT/"source"/f"{seed}_mirror_shared_dictionary.npz"
        b=ROOT/"source"/f"{seed}_native_dictionary_ids.npz"
        with np.load(a) as x,np.load(b) as y:
            for key in ("base_initialization","stream_ids","basis_dictionary","stream_codes","writable_codes"):
                assert np.array_equal(x[key],y[key])


def test_online_metric_replay():
    rows=json.loads((ROOT/"source"/"development.json").read_text())
    for row in rows:
        base,dictionary,seqs=make_world(row["seed"]);method=row["method"]
        with np.load(ROOT/"source"/f"{row['seed']}_{method}.npz") as payload:
            if method=="full_ttt_linear":
                replay=[online_full(base,s)[1:] for s in seqs]
            elif method=="fixed_shared_rank4":
                replay=[online_lowrank(base,dictionary[:,:RANK],s)[1:] for s in seqs]
            elif method=="independent_rank4":
                replay=[online_lowrank(base,dictionary[:,s["mode"]*RANK:(s["mode"]+1)*RANK],s)[1:] for s in seqs]
            else:
                codes=payload["stream_codes"]
                replay=[online_lowrank(base,dictionary[:,int(c)*RANK:(int(c)+1)*RANK],s)[1:] for c,s in zip(codes,seqs)]
        replay=np.asarray(replay)
        assert abs(float(replay[:,0].mean())-row["mean_prequential_nll"])<1e-10
        assert abs(float(replay[:,1].mean())-row["mean_token_mse"])<1e-10
        assert abs(float(replay[:,2].mean())-row["mean_final32_mse"])<1e-10
