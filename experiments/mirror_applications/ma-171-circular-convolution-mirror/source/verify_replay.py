import csv, json, math, sys
from collections import defaultdict
from pathlib import Path
import torch
from model import METHODS, compute_proxy
from run import D,O,ROLES,UPDATES,BATCH,evaluate,fit,world

def main(core, exploratory, devgrid):
    torch.set_num_threads(1)
    all_rows=[]
    for path in (core,exploratory,devgrid):
        all_rows.extend(csv.DictReader(open(path,newline='')))
    unique=[];seen={}
    for row in all_rows:
        key=(row['seed'],row['condition'],row['method'],row['learning_rate'])
        if key in seen:
            old=seen[key]
            for field in ('serialized_bytes','activation_mse','r2','active_compute_proxy'):
                assert row[field]==old[field],(key,field,row[field],old[field])
            continue
        seen[key]=row;unique.append(row)
    max_mse=max_r2=0.0
    for row in unique:
        seed=int(row['seed']);condition=row['condition'];ci=0 if condition=='aligned_hrr' else 1;mi=METHODS.index(row['method'])
        lr=float(row['learning_rate'])
        w=world(seed,condition);ds=seed+10000+ci*5000
        m,_=fit(row['method'],w,condition,seed+mi*73+ci*10000,ds,lr)
        mse,r2,_,_=evaluate(m,w,condition,seed+91+ci*3000)
        assert int(row['serialized_bytes'])==m.serialized_payload_bytes(),row
        assert int(row['active_compute_proxy'])==compute_proxy(row['method'],UPDATES*BATCH),row
        assert int(row['train_examples'])==UPDATES*BATCH and int(row['optimizer_updates'])==UPDATES,row
        dm=abs(float(row['activation_mse'])-mse);dr=abs(float(row['r2'])-r2)
        max_mse=max(max_mse,dm);max_r2=max(max_r2,dr)
        assert math.isclose(float(row['activation_mse']),mse,rel_tol=1e-6,abs_tol=1e-10),(row,mse)
        assert math.isclose(float(row['r2']),r2,rel_tol=1e-6,abs_tol=1e-8),(row,r2)
    print(json.dumps({'experiment_id':'MA-171','report_records':len(all_rows),'distinct_training_rows_replayed':len(unique),'payload_bytes_and_compute_exact':True,'max_abs_mse_difference':max_mse,'max_abs_r2_difference':max_r2,'development_candidate_records':sum(r['split']=='development_candidate' for r in all_rows),'exploratory_wrong_lr_rows':sum(r['split']=='exploratory_wrong_lr' for r in all_rows),'fresh_a1_rows':sum(r['split']=='fresh_a1' for r in all_rows)},indent=2))

if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
