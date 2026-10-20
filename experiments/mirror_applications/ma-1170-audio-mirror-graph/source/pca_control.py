"""Additional *posthoc* baseline: train-only rank-16 PCA, 8-bit signed feature code.
It is not part of the preregistered MA-1170 decision gate.
"""
import io,json,time,statistics
from pathlib import Path
import torch
from audio import main_data

@torch.no_grad()
def nmse(a,b):return float(((a-b)**2).sum()/((b**2).sum()+1e-12))

if __name__=='__main__':
    torch.set_num_threads(1)
    train,mu,sd,meta=main_data()
    # PCA centered on standardized training data (per-channel mean=0 by construction).
    c=train['train'].mean(0)
    x=train['train']-c
    _,S,Vh=torch.linalg.svd(x,full_matrices=False)
    V=Vh[:16].T.contiguous()
    ztrain=x@V
    qmin=torch.quantile(ztrain,0.005,dim=0)
    qmax=torch.quantile(ztrain,0.995,dim=0)
    qscale=(qmax-qmin).clamp_min(1e-5)/255
    data={}
    for key,d in train.items():
        z=(d-c)@V
        reconstruction=z@V.T+c
        qz=torch.round((z-qmin)/qscale).clamp(0,255).to(torch.uint8)
        rz=(qz.float()*qscale+qmin)@V.T+c
        data[key]={'nmse':nmse(reconstruction,d),'u8_nmse':nmse(rz,d)}
    packet=io.BytesIO();torch.save({'pca_basis':V,'mean':c,'quant_min':qmin,'quant_scales':qscale,'train_mel_mu':mu,'train_mel_sd':sd},packet)
    xs=train['fresh'][:1]
    for _ in range(20): y=(((xs-c)@V)@V.T+c)
    timings=[]
    for _ in range(60):
        t0=time.perf_counter_ns();y=(((xs-c)@V)@V.T+c);timings.append((time.perf_counter_ns()-t0)/1000)
    result={'label':'EXPLORATORY POSTHOC rank16 PCA (outside prereg MA-1170 acceptance)',
            'per_split':data,'n_train':len(train['train']),'pca_rank':16,'parameters_as_basis_and_mu':int(V.numel()+c.numel()),
            'packet_bytes':len(packet.getvalue()),'fresh_u8_code_bytes':12+len(train['fresh'])*16,
            'fresh_fp32_feature_bytes':len(train['fresh'])*64*4,'b1_median_us':statistics.median(timings),
            'coding':'8-bit percentile [0.5%,99.5%] signed, extra two train-fitted scalars per coordinate charged',
            'note':'PCA used only train, no hyperparameter fit to fresh data; included to avoid weak AE baselines.'}
    out=Path(__file__).resolve().parents[1]/'results'/'EXPLORATORY_PCA16.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
