"""Fresh independent 3-world test for sparse directions (frozen beforehand)."""
from __future__ import annotations
import csv,io,os,time,platform,statistics
from pathlib import Path
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from model import N,NativeMLP,weight_to_flat,invert_nonlinear_orbit,seed_transform
from run import ROOT,RESULTS,baseline,evaluate,fit_weights
from dev_direction_search import AddressedMirrorMLP

WORLDS=(618310,618311,618312)
SEEDS=(11,12)
MS=(512,768)


def data_world(world):
    raw=load_digits()
    xx=torch.as_tensor(raw.images.reshape(-1,64).astype(np.float64)/16.0,dtype=torch.float64)
    yy=torch.as_tensor(raw.target,dtype=torch.int64)
    ids=np.arange(len(yy))
    work,test=train_test_split(ids,test_size=.20,stratify=yy.numpy(),random_state=world)
    train,dev=train_test_split(work,test_size=.20,stratify=yy.numpy()[work],random_state=world+1000)
    return xx,yy,{'train':train.tolist(),'dev':dev.tolist(),'test':test.tolist()}


def run():
    torch.set_num_threads(1)
    out=RESULTS/'direction_fresh_replication.csv'
    if out.exists():raise FileExistsError(out)
    rows=[]
    for world in WORLDS:
        xx,yy,split=data_world(world)
        for seed in SEEDS:
            teacher,teacher_metrics=baseline(seed,xx,yy,split)
            if teacher_metrics['dev_accuracy']<.90:
                raise RuntimeError(f'native viability FAIL world {world} seed {seed}')
            native_acc,native_nll=evaluate(teacher,xx,yy,split,'test')
            target=weight_to_flat(teacher)
            for m in MS:
                for mode in ('plain_top','mirror_top','mirror_first'):
                    nonlinear=mode.startswith('mirror')
                    inv=invert_nonlinear_orbit(target) if nonlinear else seed_transform(target,inverse=True)
                    if mode.endswith('top'):
                        index=torch.topk(inv.abs(),m).indices.sort().values
                    else:
                        index=torch.arange(m)
                    model=AddressedMirrorMLP(index,nonlinear=nonlinear)
                    with torch.no_grad():model.coords.copy_(inv[index.long()])
                    fit=fit_weights(model,target)
                    acc,nll=evaluate(model,xx,yy,split,'test')
                    with torch.no_grad(): l2=float(torch.linalg.vector_norm(model.weights()-target))
                    b=model.serialized_bytes()
                    row={'world':world,'seed':seed,'m':m,'mode':mode,
                         'native_dev_accuracy':teacher_metrics['dev_accuracy'],
                         'native_test_accuracy':native_acc,'native_test_nll':native_nll,
                         'reconstructed_test_accuracy':acc,'reconstructed_test_nll':nll,
                         'accuracy_drop_pp':(native_acc-acc)*100.,
                         'weight_l2':l2,'payload_bytes':b,'index_bytes_nominal':2*m,
                         'optimizer_calls':fit['optimizer_iterations'],'fit_seconds':fit['fit_seconds']}
                    rows.append(row)
                    print('FRESH',world,seed,m,mode,'baseline',round(native_acc,4),
                          'reconstructed',round(acc,4),'L2',round(l2,3),'bytes',b,flush=True)
                    with out.open('w',newline='') as f:
                        wr=csv.DictWriter(f,list(rows[0]));wr.writeheader();wr.writerows(rows)

if __name__=='__main__':run()
