"""Post-run accounting correction: charge per-world train-only input normalization state."""
import hashlib,io,json,csv
from pathlib import Path
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from experiment import split_world
ROOT=Path(__file__).parent
WORLDS=(79010,79021,79022,79023,79110,79121,79122,79123,79210,79221,79222,79223,79321,79322)

def build(world):
    ds=load_digits();idx=np.arange(len(ds.target))
    tr,rest=train_test_split(idx,train_size=0.60,random_state=world,stratify=ds.target)
    va,te=train_test_split(rest,train_size=0.50,random_state=world+20000,stratify=ds.target[rest])
    mu=ds.data[tr].mean(axis=0)/16
    sd=np.maximum(ds.data[tr].std(axis=0)/16,0.10)
    state={'mu':torch.tensor(mu,dtype=torch.float64),'sd':torch.tensor(sd,dtype=torch.float64),
           'image_divisor':16,'shape':[8,8],'schema':'train_only_per_pixel_standardization_v1'}
    buf=io.BytesIO();torch.save(state,buf)
    contents=buf.getvalue(); size=len(contents)
    split=split_world(world)
    for name,ii in [('train',tr),('val',va),('test',te)]:
        x=torch.tensor((ds.data[ii]/16-mu)/sd,dtype=torch.float32)
        assert torch.equal(x,split[name][0]),(world,name)
    return contents,{'world':world,'preprocessing_payload_bytes':size,'sha256':hashlib.sha256(contents).hexdigest(),
         'split_index_hash':split['split_index_sha256'],'match_all_split_features':True}

def run():
    rows=[]
    for w in WORLDS:
        payload,row=build(w)
        fname=ROOT/'models'/f'preprocessing_world{w}.pt';fname.write_bytes(payload)
        rows.append(row)
    with (ROOT/'results/PREPROCESSING_BYTES.csv').open('w',newline='') as file:
        ww=csv.DictWriter(file,fieldnames=rows[0].keys());ww.writeheader();ww.writerows(rows)
    print('audited worlds',len(rows),'preprocessor bytes',sorted(set(r['preprocessing_payload_bytes'] for r in rows)))
if __name__=='__main__':run()
