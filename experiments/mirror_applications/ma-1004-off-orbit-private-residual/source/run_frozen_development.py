#!/usr/bin/env python3
"""Evaluate frozen MACE-MP on locked MA-1004 support/development rows only."""
import argparse, json, pathlib, time, torch
from ase import Atoms
from mace.calculators import mace_mp
KCAL_MOL_TO_EV = 0.0433641153087705

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True); ap.add_argument('--out',required=True); ap.add_argument('--shots',type=int,default=32); a=ap.parse_args()
    torch.set_num_threads(1)
    root=pathlib.Path(a.data_dir); exp=pathlib.Path(__file__).resolve().parents[1]
    manifest=json.loads((exp/'runs/ROW_MANIFEST.json').read_text())
    ur=next(x for x in manifest['splits'] if x['molecule']=='uracil' and x['role']=='target_support')
    dv=next(x for x in manifest['splits'] if x['molecule']=='uracil' and x['role']=='development')
    data=torch.load(root/'uracil.pt',map_location='cpu',weights_only=False)
    z=data['z'].tolist(); ptr=data['ptr']; pos=data['pos']; E=data['energy']; F=data['forces']
    calc=mace_mp(model='small',device='cpu',default_dtype='float64')
    def predict(row):
        lo=int(ptr[row]); hi=int(ptr[row+1]); atoms=Atoms(numbers=z,positions=pos[lo:hi].numpy()); atoms.calc=calc
        ts=time.perf_counter(); en=atoms.get_potential_energy(); fo=atoms.get_forces(); return en,fo,time.perf_counter()-ts,lo,hi
    start=time.perf_counter(); support=[]; support_forces=[]
    for r in ur['selected_rows'][:a.shots]:
        en,fo,dt,lo,hi=predict(r['row']); gold_f=F[lo:hi].numpy()*KCAL_MOL_TO_EV; support.append((en,float(E[r['row']])*KCAL_MOL_TO_EV)); support_forces.append((fo,gold_f))
    offset=sum(y-x for x,y in support)/len(support)
    num=sum(float((pred*gold).sum()) for pred,gold in support_forces); den=sum(float((pred*pred).sum()) for pred,gold in support_forces); force_scale=num/den
    scaled_offset=sum(y-force_scale*x for x,y in support)/len(support)
    es=[]; fs=[]; gate_es=[]; gate_fs=[]; infer=[]; details=[]
    for r in dv['selected_rows']:
        en,fo,dt,lo,hi=predict(r['row']); gold_e=float(E[r['row']])*KCAL_MOL_TO_EV
        gold_f=F[lo:hi].numpy()*KCAL_MOL_TO_EV
        es.append(abs((en+offset-gold_e)/len(z))); fs.extend(abs((fo-gold_f)).ravel().tolist()); gate_es.append(abs((force_scale*en+scaled_offset-gold_e)/len(z))); gate_fs.extend(abs((force_scale*fo-gold_f)).ravel().tolist()); infer.append(dt)
        details.append({'old_index':r['old_index'],'row':r['row'],'energy_abs_eV_atom':es[-1],'force_mae_eV_A':float(abs(fo-gold_f).mean()),'gate_energy_abs_eV_atom':gate_es[-1],'gate_force_mae_eV_A':float(abs(force_scale*fo-gold_f).mean()),'infer_s':dt})
    result={'model':'MACE-MP small','role':'development only','fresh_rows_accessed':False,'support_rows':len(support),'development_rows':len(details),'energy_offset_eV':offset,'energy_mae_eV_atom':sum(es)/len(es),'force_component_mae_eV_A':sum(fs)/len(fs),'gate':'single shared scalar on total energy fit to support forces; per-task energy offset refit','gate_force_scale':force_scale,'gate_energy_offset_eV':scaled_offset,'gate_energy_mae_eV_atom':sum(gate_es)/len(gate_es),'gate_force_component_mae_eV_A':sum(gate_fs)/len(gate_fs),'mean_inference_s_per_structure':sum(infer)/len(infer),'wall_clock_s':time.perf_counter()-start,'details':details}
    out=pathlib.Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))
if __name__=='__main__':main()
