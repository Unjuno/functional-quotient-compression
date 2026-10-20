"""Posthoc causal edge-ranking audit. Exploratory, not part of frozen acceptance gates."""
import csv,sys
from pathlib import Path
import torch
import torch.nn.functional as F
from model import AudioAE
from audio import main_data


def decode_masked(model,z,off):
    w=model.graph.weight_matrices().reshape(16,4,4).clone()
    w[off]=0
    w=w.reshape(4,4,4,4)
    state=z.reshape(-1,4,4)
    for _ in range(2):
        state=F.relu(state+0.5*torch.einsum('dsij,bsj->bdi',w,state))
    return model.decoder(state.reshape(-1,16))

def nmse(y,x):return float((y-x).square().sum()/(x.square().sum()+1e-16))

if __name__=='__main__':
    root=Path(sys.argv[1])
    dat,*_=main_data(); x=dat['fresh']
    rows=[]
    for seed in [31,37,41]:
        for kind in ['mirror','affine']:
            ck=torch.load(root/'checkpoints'/f'{kind}_{seed}.pt',weights_only=True)
            model=AudioAE(kind);model.load_state_dict(ck['weights']);model.eval()
            with torch.no_grad():
                z=model.encode(x)
                order=torch.argsort(torch.sigmoid(model.graph.gates))
                rand=torch.randperm(16,generator=torch.Generator().manual_seed(1234+seed))[:8]
                args={'all':torch.tensor([],dtype=torch.long),'bottom8':order[:8], 'top8':order[-8:],'random8':rand}
                errs={k:nmse(decode_masked(model,z,off),x) for k,off in args.items()}
            rows.append({'seed':seed,'kind':kind,**{k+'_nmse':v for k,v in errs.items()},
                         'low_vs_high_delta_nmse':errs['top8']-errs['bottom8']})
    with open(root/'results'/'EXPLORATORY_EDGE_MASK.csv','w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0].keys()));wr.writeheader();wr.writerows(rows)
    for r in rows:print(r)
