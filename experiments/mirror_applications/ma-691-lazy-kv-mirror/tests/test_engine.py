import math, sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from engine import row_rot, attention, run_seed

def test_lazy_exact_small():
    rows,checks=run_seed(69101)
    assert max(r['max_abs_error'] for r in rows) <= 2e-6

def test_rope_commutes_for_same_planes():
    _,c=run_seed(69102)
    assert c['rope_commutation_max_abs_error'] <= 2e-6

def test_mla_absorption_exact():
    _,c=run_seed(69103)
    assert c['mla_absorption_max_abs_error'] <= 2e-6

def test_latent_cache_is_smaller():
    _,c=run_seed(69101)
    assert c['mla_latent_cache_bytes'] < c['mla_full_kv_bytes']
