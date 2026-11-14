#!/usr/bin/env python3
"""PA300 EquiSparseDeltaSTR raw-model compatibility probe; no training or metrics."""
import argparse, json, pathlib, sys, time, torch
from ase import Atoms
from mace.calculators import MACECalculator, mace_mp
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from pa300_sparse_delta_core import EquiSparseDeltaSTR

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(1)
    raw=mace_mp(model='small',device='cpu',return_raw_model=True,default_dtype='float64')
    calc=MACECalculator(models=raw,device='cpu',default_dtype='float64')
    d=torch.load(a.data,map_location='cpu',weights_only=False)
    atoms=Atoms(numbers=d['z'].tolist(),positions=d['pos'][:12].numpy())
    batch=calc._atoms_to_batch(atoms).to_dict();batch['positions'].requires_grad_(True)
    t=time.perf_counter();model=EquiSparseDeltaSTR(raw,init_threshold=1e-4)
    n=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())
    out=model(batch,training=True,compute_force=True,compute_virials=False,compute_stress=False)
    result={'control':'official PA300 EquiSparseDeltaSTR','compatibility':'forward pass completed','prior':'MACE-MP small','trainable_parameter_count':n,'wrapped_total_parameter_count':total,'force_output_shape':list(out['forces'].shape),'force_output_requires_grad':bool(out['forces'].requires_grad),'initial_sparsity':model.get_sparsity(),'wrap_plus_forward_seconds':time.perf_counter()-t,'training_updates':0,'quality_metrics_computed':False}
    pathlib.Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
