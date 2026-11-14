#!/usr/bin/env python3
"""MA-1003 fixed CPU screen: shared readout basis, native code, independent readout."""
import argparse, json, pathlib, time, hashlib, io, zipfile
import numpy as np, torch, requests
from ase import Atoms
from mace.calculators import MACECalculator

EV_J=1.602176634e-19
ANG_M=1.0e-10
FORCE_N_PER_EV_A=EV_J/ANG_M
FACETS=['cu100','cu110','cu111','cu211']

def load_records(data_dir, roles):
    root=pathlib.Path(data_dir); idx=json.load(open(root/'download_manifest.json')); out=[]
    for r in idx:
        if r['role'] not in roles: continue
        p=pathlib.Path(r['path']); d=json.load(open(p))['data']['archive']['run'][0]
        c=d['calculation'][0]; s=d['system'][0]; a=s['atoms']; n=len(a['species'])
        if n != 56 or c.get('calculation_converged') is not True: continue
        pos=np.asarray(a['positions'],dtype=np.float64)/ANG_M
        cell=np.asarray(a['lattice_vectors'],dtype=np.float64)/ANG_M
        e=float(c['energy']['total']['value'])/EV_J
        f=np.asarray(c['forces']['free']['value'],dtype=np.float64)/FORCE_N_PER_EV_A
        if not (np.isfinite(pos).all() and np.isfinite(cell).all() and np.isfinite(e) and np.isfinite(f).all()): continue
        atoms=Atoms(numbers=a['species'],positions=pos,cell=cell,pbc=[True,True,True])
        # geometry key is solely positions/cell/species, never labels/metrics.
        geo=hashlib.sha256(np.round(pos,6).astype('<f8').tobytes()+np.round(cell,6).astype('<f8').tobytes()+np.asarray(a['species'],dtype='<i4').tobytes()).hexdigest()
        out.append({**{k:r[k] for k in ['role','stage','facet','run_id','entry_id']},'atoms':atoms,'energy':e,'forces':f,'geometry_hash':geo})
    return out

def descriptor_rows(records, model_path):
    calc=MACECalculator(model_paths=model_path,device='cpu',default_dtype='float32')
    model=calc.models[0]; rows=[]; start=time.perf_counter()
    for r in records:
        b=calc._atoms_to_batch(r['atoms']).to_dict();b['positions'].requires_grad_(True)
        with torch.enable_grad(): out=model(b,compute_force=True,training=True)
        z=out['node_feats'].detach().sum(0).cpu().numpy().astype(np.float64)
        eb=float(out['energy'].detach().cpu().reshape(-1)[0])
        fb=out['forces'].detach().cpu().numpy().astype(np.float64)
        rows.append({**{k:r[k] for k in ['role','stage','facet','run_id','entry_id','energy','forces','geometry_hash']},'atoms':r['atoms'],'z':z,'ebase':eb,'fbase':fb})
        del out,b
    print('feature_rows',len(rows),'feature_wall_s',round(time.perf_counter()-start,3),flush=True)
    return rows,calc

def unique_role(rows,role):return [r for r in rows if r['role']==role]
def stage(rows,name):return [r for r in rows if r['stage']==name]
def ridge_fit(X,y,alpha):
    X=np.asarray(X);y=np.asarray(y);return np.linalg.solve(X.T@X+alpha*np.eye(X.shape[1]),X.T@y)
def predict_features(row,mu,scale):return np.r_[(row['z']-mu)/scale,1.0]
def fit_surface_readouts(rows, mu, scale, offset, alpha=1e-2):
    weights={}
    for facet in ['cu100','cu110','cu111']:
        rr=[r for r in rows if r['facet']==facet]
        if not rr:continue
        X=np.stack([predict_features(r,mu,scale) for r in rr]);y=np.array([r['energy']-r['ebase']-offset for r in rr])
        weights[facet]=ridge_fit(X,y,alpha)
    return weights

def factor_basis(weights,rank):
    W=np.stack(list(weights.values()));u,s,vt=np.linalg.svd(W,full_matrices=False)
    B=vt[:rank].T*s[:rank]
    return B

def code_fit(support,B,mu,scale,offset,alpha=1e-3):
    X=np.stack([predict_features(r,mu,scale) for r in support]); y=np.array([r['energy']-r['ebase']-offset for r in support])
    A=X@B;return np.linalg.solve(A.T@A+alpha*np.eye(B.shape[1]),A.T@y)

def independent_fit(support,mu,scale,offset,alpha=1e-2):
    X=np.stack([predict_features(r,mu,scale) for r in support]); y=np.array([r['energy']-r['ebase']-offset for r in support])
    return ridge_fit(X,y,alpha)

def metrics_energy(rows,weights,mu,scale,offset):
    err=[];diff=[]
    for r in rows:
        w=weights.get(r['facet'])
        corr=0. if w is None else predict_features(r,mu,scale)@w
        ehat=r['ebase']+offset+corr
        err.append((ehat-r['energy'])/56.0)
        diff.append((ehat-r['energy']))
    return float(np.mean(np.abs(err))),float(np.mean(np.abs(diff)))

def eval_forces(rows,weights,mu,scale,offset,calc):
    model=calc.models[0]; allerr=[]; start=time.perf_counter()
    for r in rows:
        b=calc._atoms_to_batch(r['atoms']).to_dict();b['positions'].requires_grad_(True)
        with torch.enable_grad(): out=model(b,compute_force=True,training=True)
        base=out['forces'].detach().cpu().numpy()
        w=weights.get(r['facet'])
        if w is None: pred=base
        else:
            wt=torch.as_tensor(w[:-1]/scale,dtype=out['node_feats'].dtype)
            de=(out['node_feats'].sum(0)*wt).sum()
            grad=torch.autograd.grad(de,b['positions'],retain_graph=False)[0].detach().cpu().numpy()
            pred=base-grad
        allerr.extend((pred-r['forces']).reshape(-1).tolist())
        del out,b
    return float(np.mean(np.abs(allerr))),time.perf_counter()-start

def save_payload(path,B,codes,mu,scale,offset,kind):
    buf=io.BytesIO();np.savez_compressed(buf,B=np.asarray(B,dtype=np.float32),codes=np.asarray(codes,dtype=np.float32),feature_mean=np.asarray(mu,dtype=np.float32),feature_scale=np.asarray(scale,dtype=np.float32),energy_offset=np.asarray([offset],dtype=np.float64),format=np.asarray([kind]))
    data=buf.getvalue();pathlib.Path(path).write_bytes(data);return len(data),hashlib.sha256(data).hexdigest(),data

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',required=True);ap.add_argument('--model',required=True);ap.add_argument('--out',required=True);ap.add_argument('--mode',choices=['dev','audit'],required=True);a=ap.parse_args()
    torch.set_num_threads(1);t0=time.perf_counter();records=load_records(a.data_dir,{'basis_fit','hyperparameter_development','few_shot_support'} if a.mode=='dev' else {'fresh_audit'})
    rows,calc=descriptor_rows(records,a.model);print('role_counts', {x:sum(r['role']==x for r in rows) for x in set(r['role'] for r in rows)},flush=True)
    fit=[r for r in rows if r['stage']=='2_initial_data_0_MD'];
    if not fit: raise SystemExit('no converged stage-2 Cu54H2 basis-fit data')
    mu=np.mean(np.stack([r['z'] for r in fit]),axis=0);scale=np.std(np.stack([r['z'] for r in fit]),axis=0);scale=np.where(scale<1e-8,1.0,scale)
    offset=float(np.mean([r['energy']-r['ebase'] for r in fit]))
    taskw=fit_surface_readouts(fit,mu,scale,offset)
    dev3=[r for r in rows if r['stage']=='3_new_datapoints_1'];dev4=[r for r in rows if r['stage']=='4_new_datapoints_2']
    # Selection only uses stage 3/4 development energy MAE, with exact geometry overlaps excluded.
    combinations=[]
    for rank in [1,2,3]:
        B=factor_basis(taskw,rank)
        for k in [1,2,4,8]:
            errs=[]
            for support_rows,eval_rows in [(dev3,dev4),(dev4,dev3)]:
                for facet in FACETS:
                    sup=sorted([r for r in support_rows if r['facet']==facet],key=lambda r:r['entry_id'])[:k]
                    sh={r['geometry_hash'] for r in sup}
                    ev=[r for r in eval_rows if r['facet']==facet and r['geometry_hash'] not in sh]
                    if not sup or not ev:continue
                    m=code_fit(sup,B,mu,scale,offset);w=B@m
                    errs.extend([abs((r['ebase']+offset+predict_features(r,mu,scale)@w-r['energy'])/56.0) for r in ev])
            combinations.append({'rank':rank,'shots':k,'dev_energy_mae_eV_atom':float(np.mean(errs)) if errs else None})
    choices=[x for x in combinations if x['dev_energy_mae_eV_atom'] is not None]
    best=min(choices,key=lambda x:(x['dev_energy_mae_eV_atom'],x['rank'],x['shots']))
    B=factor_basis(taskw,best['rank'])
    outdir=pathlib.Path(a.out);outdir.mkdir(parents=True,exist_ok=True)
    results={'mode':a.mode,'data_records_after_filter':len(rows),'offset_eV':offset,'features':256,'basis_source_facets':list(taskw),'dev_grid':combinations,'selected':best,'protocol':'MA-1003 PROTOCOL.json plus Amendments 01-02','wall_feature_and_fit_s':time.perf_counter()-t0}
    if a.mode=='dev':
        for direction,support_rows,eval_rows in [('stage3_to_stage4',dev3,dev4),('stage4_to_stage3',dev4,dev3)]:
            mir={};ind={};
            for facet in FACETS:
                sup=sorted([r for r in support_rows if r['facet']==facet],key=lambda r:r['entry_id'])[:best['shots']]
                if not sup:continue
                mir[facet]=B@code_fit(sup,B,mu,scale,offset)
                ind[facet]=independent_fit(sup,mu,scale,offset)
            ev=[r for r in eval_rows if r['geometry_hash'] not in {s['geometry_hash'] for s in support_rows}]
            em=metrics_energy(ev,mir,mu,scale,offset);ei=metrics_energy(ev,ind,mu,scale,offset)
            fm,tm=eval_forces(ev,mir,mu,scale,offset,calc);fi,ti=eval_forces(ev,ind,mu,scale,offset,calc)
            frozen={f:np.zeros(257) for f in FACETS};ef=metrics_energy(ev,frozen,mu,scale,offset)
            ff=float(np.mean(np.abs(np.concatenate([(r['fbase']-r['forces']).reshape(-1) for r in ev]))))
            results.setdefault('dev_selected_metrics',[]).append({'direction':direction,'frozen_energy_mae_eV_atom':ef[0],'frozen_energy_absolute_mae_eV':ef[1],'frozen_force_component_mae_eV_A':ff,'mirror_energy_mae_eV_atom':em[0],'mirror_energy_absolute_mae_eV':em[1],'mirror_force_component_mae_eV_A':fm,'independent_energy_mae_eV_atom':ei[0],'independent_energy_absolute_mae_eV':ei[1],'independent_force_component_mae_eV_A':fi,'mirror_force_eval_wall_s':tm,'independent_force_eval_wall_s':ti,'n_eval':len(ev),'mirror_adapter_energy_mac_per_structure':256*best['rank']+best['rank'],'independent_adapter_energy_mac_per_structure':256})
        codes=[];ind_weights=[]
        for facet in FACETS:
            sup=sorted([r for r in rows if r['stage']=='5_new_datapoints_3' and r['facet']==facet],key=lambda r:r['entry_id'])[:best['shots']]
            if sup:codes.append(code_fit(sup,B,mu,scale,offset));ind_weights.append(independent_fit(sup,mu,scale,offset))
        mbytes,mhash,payload=save_payload(outdir/'mirror_payload.npz',B,codes,mu,scale,offset,'shared_low_rank_readout_v1')
        (outdir/'native_factorized_payload.npz').write_bytes(payload)
        nbytes,nhash=mbytes,mhash
        ibytes,ihash,_=save_payload(outdir/'independent_payload.npz',np.zeros((257,0)),ind_weights,mu,scale,offset,'independent_readouts')
        results['payloads']={'mirror_bytes':mbytes,'mirror_sha256':mhash,'native_factorized_bytes':nbytes,'native_factorized_sha256':nhash,'independent_bytes':ibytes,'independent_sha256':ihash,'byte_ratio_mirror_to_independent':mbytes/ibytes,'native_alias_byte_equal':mbytes==nbytes,'native_alias_payload_hash_equal':mhash==nhash}
        np.savez_compressed(outdir/'selected_weights.npz',B=B,mu=mu,scale=scale,offset=offset,rank=best['rank'],shots=best['shots'],mirror_codes=np.array(codes,dtype=object),independent=np.array(ind_weights,dtype=object))
    (outdir/'screen.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
    print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__':main()
