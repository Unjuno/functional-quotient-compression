"""MA-1171: CPU synthetic activation SAE/virtual Mirror detector screening.

No ground-truth dictionary or support enters the training objective. The known
planted factors are used only by the offline evaluator on final checkpoints.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import statistics
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / 'PROTOCOL.json').read_text())
D, FNUM, K = 32, 64, 3
PAGE_COUNT, ADDR_DIM, RANK = 8, 6, 8
METHODS = P['algorithms']
WORLDS = P['worlds']
SEEDS = P['seeds']


def fixed_maps():
    rng = torch.Generator().manual_seed(1171)
    perms = torch.stack([torch.randperm(D, generator=rng) for _ in range(PAGE_COUNT)])
    signs = torch.stack([2 * torch.randint(0, 2, (D,), generator=rng).float() - 1 for _ in range(PAGE_COUNT)])
    keys = F.normalize(torch.randn(PAGE_COUNT, ADDR_DIM, generator=rng), dim=-1)
    v = F.normalize(torch.randn(D, generator=rng), dim=0)
    h = torch.eye(D) - 2 * torch.outer(v, v)
    return perms, signs, keys, h


def sparse_relu_topk(x):
    r = F.relu(x)
    val, idx = torch.topk(r, K, dim=-1)
    return torch.zeros_like(r).scatter(-1, idx, val)


class SAE(nn.Module):
    def __init__(self, method):
        super().__init__()
        assert method in METHODS
        self.method = method
        self.decoder_raw = nn.Parameter(torch.randn(FNUM, D) * (D ** -0.5))
        self.outbias = nn.Parameter(torch.zeros(D))
        self.enc_bias = nn.Parameter(torch.zeros(FNUM))
        if method.startswith('dense'):
            self.enc_raw = nn.Parameter(torch.randn(FNUM, D) * (D ** -0.5))
        elif method == 'lowrank_loop':
            self.factor_a = nn.Parameter(torch.randn(FNUM, RANK) * 0.22)
            self.factor_b = nn.Parameter(torch.randn(RANK, D) * 0.2)
            self.gain = nn.Parameter(torch.zeros(FNUM))
        else:
            self.core = nn.Parameter(torch.randn(D) * 0.7)
            self.address = nn.Parameter(torch.randn(FNUM, ADDR_DIM) * 0.5)
            self.gain = nn.Parameter(torch.zeros(FNUM))
            pm, sg, keys, h = fixed_maps()
            self.register_buffer('perms', pm, persistent=False)
            self.register_buffer('signs', sg, persistent=False)
            self.register_buffer('keys', keys, persistent=False)
            self.register_buffer('householder', h, persistent=False)

    def matrices(self):
        d = F.normalize(self.decoder_raw, dim=-1)
        if self.method.startswith('dense'):
            e = F.normalize(self.enc_raw, dim=-1)
        elif self.method == 'lowrank_loop':
            e = self.factor_a @ self.factor_b
            e = F.normalize(e, dim=-1) * (F.softplus(self.gain) + 0.1)[:, None]
        else:
            pages = self.core[self.perms] * self.signs
            mix = torch.softmax((self.address @ self.keys.T) * 2.5, dim=-1)
            v = mix @ pages
            if self.method == 'mirror_loop':
                v = (torch.tanh(2.5 * (v @ self.householder)) @ self.householder.T) / 2.5
            e = F.normalize(v, dim=-1) * (F.softplus(self.gain) + 0.1)[:, None]
        return e, d

    def encode_and_reconstruct(self, x):
        e, d = self.matrices()
        z = sparse_relu_topk(x @ e.T + self.enc_bias)
        if self.method != 'dense_onepass':
            for _ in range(2):
                residual = x - (z @ d + self.outbias)
                z = sparse_relu_topk(z + 0.4 * (residual @ e.T))
        return z, z @ d + self.outbias

    def forward(self, x):
        return self.encode_and_reconstruct(x)[1]


def ground_truth(seed, world):
    g = torch.Generator().manual_seed(seed * 1000 + (10000 if world == 'iid' else 20000))
    dictionary = F.normalize(torch.randn(FNUM, D, generator=g), dim=-1)
    return dictionary


def generate(seed, world, split, n=None):
    if n is None:
        n = {'train':8192, 'dev':1024, 'fresh_id':2048, 'fresh_ood':2048}[split]
    code = {'train':0, 'dev':1, 'fresh_id':2, 'fresh_ood':3}[split]
    g = torch.Generator().manual_seed(seed * 100 + code + (100000 if world == 'correlated_pairs' else 0))
    if world == 'correlated_pairs' and split != 'fresh_ood':
        anchor = torch.randint(0, FNUM, (n,), generator=g)
        partner = anchor ^ 1
        replace = torch.rand(n, generator=g) >= 0.75
        partner[replace] = torch.randint(0, FNUM, (int(replace.sum()),), generator=g)
        third = torch.randint(0, FNUM, (n,), generator=g)
        ids = torch.stack([anchor, partner, third], -1)
        for j in [1, 2]:
            conflict = (ids[:, j:j+1] == ids[:, :j]).any(dim=-1)
            while bool(conflict.any()):
                ids[conflict, j] = torch.randint(0, FNUM, (int(conflict.sum()),), generator=g)
                conflict = (ids[:, j:j+1] == ids[:, :j]).any(dim=-1)
    else:
        choices = torch.rand(n, FNUM, generator=g)
        ids = torch.topk(choices, K, dim=-1).indices
    amp = 0.65 + 0.7 * torch.rand(n, K, generator=g)
    w = ground_truth(seed, world)
    x = (w[ids] * amp[..., None]).sum(dim=1) + 0.01 * torch.randn(n, D, generator=g)
    return x, ids


def normalized_mse(p, x):
    return float(((p - x) ** 2).mean() / (x.square().mean() + 1e-12))


def true_feature_f1(z, ids, decoder, w):
    # Plant dictionary is NOT exposed to the model, gradients or dev selection.
    cos = (w @ F.normalize(decoder, dim=-1).T).detach().cpu().numpy()
    rows, columns = linear_sum_assignment(-cos)
    permutation = np.zeros(FNUM, dtype=np.int64)
    permutation[rows] = columns
    flags = (z[:, permutation] > 1e-8)
    truth = torch.zeros(z.shape[0], FNUM, dtype=torch.bool)
    truth.scatter_(1, ids, True)
    tp = int((flags & truth).sum())
    fp = int((flags & ~truth).sum())
    fn = int((~flags & truth).sum())
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return f1, float(cos[rows, columns].mean()), float(flags.float().sum(-1).mean())


def inference_packet(model):
    b = io.BytesIO()
    torch.save({'method':model.method, 'program_seed':1171, 'state':model.state_dict()}, b)
    return b.getvalue()


def timing(model, bs):
    model.eval()
    x = torch.randn(bs, D)
    values = []
    with torch.no_grad():
        for _ in range(15): model(x)
        for _ in range(100):
            start = time.perf_counter_ns()
            model(x)
            values.append((time.perf_counter_ns() - start) / 1000.0)
    return float(statistics.median(values))


def run_train():
    torch.set_num_threads(1)
    (ROOT/'checkpoints').mkdir(exist_ok=True)
    (ROOT/'results').mkdir(exist_ok=True)
    out=[]
    for world in WORLDS:
        for seed in SEEDS:
            train,_ = generate(seed,world,'train')
            dev,_=generate(seed,world,'dev')
            scale = float(train.square().mean())
            for method in METHODS:
                torch.manual_seed(seed * 777 + (0 if world=='iid' else 50000))
                model = SAE(method)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-5)
                gen=torch.Generator().manual_seed(seed * 3000 + (0 if world=='iid' else 400))
                best=float('inf'); best_step=None; best_state=None
                losses=[]
                for update in range(1,501):
                    chosen=torch.randint(0, train.shape[0], (160,),generator=gen)
                    x=train[chosen]
                    optimizer.zero_grad(set_to_none=True)
                    y=model(x)
                    loss=(y-x).square().mean()/scale
                    if not torch.isfinite(loss): raise ValueError(f'nonfinite train: {world}/{seed}/{method}/{update}')
                    loss.backward()
                    optimizer.step()
                    if update % 100 == 0:
                        with torch.no_grad():
                            model.eval()
                            nmse=normalized_mse(model(dev),dev)
                            model.train()
                        losses.append((update,nmse))
                        if nmse < best:
                            best=nmse;best_step=update
                            best_state={key:val.detach().clone() for key,val in model.state_dict().items()}
                model.load_state_dict(best_state)
                path=ROOT/'checkpoints'/f'{world}_{seed}_{method}.pt'
                by=inference_packet(model)
                path.write_bytes(by)
                rec={'world':world,'seed':seed,'method':method,'dev_NMSE':best,'best_step':best_step,
                     'train_examples':8192,'updates':500,'train_batch':160,'checkpoint_sha256':hashlib.sha256(by).hexdigest(),
                     'packet_bytes':len(by),'trainable_params':sum(p.numel() for p in model.parameters()),
                     'dev_curve':json.dumps(losses,separators=(',',':'))}
                out.append(rec)
                print('TRAIN',world,seed,method,'dev',round(best,5),'@',best_step,'bytes',len(by),flush=True)
    save_csv(ROOT/'results'/'DEV_RESULTS.csv',out)
    (ROOT/'results'/'TRAIN_COMPLETE.json').write_text(json.dumps({'mode':'train_only','trained':len(out),'fresh_never_evaluated':True,'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest()},indent=2)+'\n')


def save_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)


def run_audit():
    torch.set_num_threads(1)
    pth=ROOT/'results'/'TRAIN_COMPLETE.json'
    assert pth.exists(), 'Train complete flag required for fresh opening'
    trainmeta=json.loads(pth.read_text())
    assert trainmeta['trained']==len(WORLDS)*len(SEEDS)*len(METHODS)
    assert trainmeta['protocol_sha256']==hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest()
    devrows=list(csv.DictReader((ROOT/'results'/'DEV_RESULTS.csv').open()))
    expected={(w,s,m) for w in WORLDS for s in SEEDS for m in METHODS}
    assert expected=={(r['world'],int(r['seed']),r['method']) for r in devrows}
    out=[];check=[]
    for world in WORLDS:
        for seed in SEEDS:
            w=ground_truth(seed,world)
            x_id,ids_id=generate(seed,world,'fresh_id')
            x_ood,ids_ood=generate(seed,world,'fresh_ood')
            for method in METHODS:
                path=ROOT/'checkpoints'/f'{world}_{seed}_{method}.pt'
                bb=path.read_bytes()
                packet=torch.load(io.BytesIO(bb),weights_only=True,map_location='cpu')
                assert packet['program_seed']==1171 and packet['method']==method
                model=SAE(method);model.load_state_dict(packet['state']);model.eval()
                with torch.no_grad():
                    z_id,re_id=model.encode_and_reconstruct(x_id)
                    z_ood,re_ood=model.encode_and_reconstruct(x_ood)
                    dec=model.matrices()[1]
                    id_f1,cosine,id_l0=true_feature_f1(z_id,ids_id,dec,w)
                    ood_f1,cosine2,ood_l0=true_feature_f1(z_ood,ids_ood,dec,w)
                    assert abs(cosine-cosine2)<1e-10
                    packet2=inference_packet(model)
                    assert packet2==bb, f'checkpoint roundtrip fails: {world,seed,method}'
                    row={'world':world,'seed':seed,'method':method,'fresh_id_support_F1':id_f1,'fresh_ood_support_F1':ood_f1,
                        'fresh_id_NMSE':normalized_mse(re_id,x_id),'fresh_ood_NMSE':normalized_mse(re_ood,x_ood),
                        'dictionary_unique_cos':cosine,'fresh_id_L0':id_l0,'fresh_ood_L0':ood_l0,
                        'trainable_params':sum(p.numel() for p in model.parameters()),'inference_packet_bytes':len(bb),
                        'runtime_b1_us':timing(model,1),'runtime_b128_us':timing(model,128),
                        'packet_sha256':hashlib.sha256(bb).hexdigest()}
                out.append(row);check.append({'world':world,'seed':seed,'method':method,'byte_exact_replay':True})
                print('AUDIT',world,seed,method,'F1',round(id_f1,4),round(ood_f1,4),'NMSE',round(row['fresh_id_NMSE'],4),flush=True)
    save_csv(ROOT/'RESULTS_CORE.csv',out)
    maps={(r['world'],r['seed'],r['method']):r for r in out}
    pair=[]
    for world in WORLDS:
        for seed in SEEDS:
            mirror=maps[(world,seed,'mirror_loop')]
            affine=maps[(world,seed,'affine_loop')]
            dense=maps[(world,seed,'dense_loop')]
            delta_aff=mirror['fresh_ood_support_F1']-affine['fresh_ood_support_F1']
            delta_dense=mirror['fresh_ood_support_F1']-dense['fresh_ood_support_F1']
            quality_ratio=mirror['fresh_id_NMSE']/affine['fresh_id_NMSE']
            pair.append({'world':world,'seed':seed,'delta_oodF1_vs_affine':delta_aff,'delta_oodF1_vs_dense':delta_dense,
                         'ID_NMSE_ratio_to_affine':quality_ratio,'pair_pass': delta_aff>=0.05 and delta_dense>=0.05 and quality_ratio<=1.10})
    save_csv(ROOT/'results'/'PAIRED_GATES.csv',pair)
    n_f1=sum(p['delta_oodF1_vs_affine']>=.05 and p['delta_oodF1_vs_dense']>=.05 for p in pair)
    nmse_ok=all(p['ID_NMSE_ratio_to_affine']<=1.10 for p in pair)
    verdict='PASS' if n_f1>=5 and nmse_ok else 'FAIL'
    ver={'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.json').read_bytes()).hexdigest(),
         'experiment_id':'MA-1171','worlds':WORLDS,'seeds':SEEDS,'methods':METHODS,'trained_models':len(out),'checkpoint_replay_pass':len(check),'checkpoint_replay_fail':0,
         'frozen_gate_ood_f1_pair_passes':n_f1,'frozen_gate_quality_all_pass':nmse_ok,'verdict':verdict,'timestamp':'2026-10-20 JST','eval_used_for_tuning':False,
         'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cpu':open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].split(':')[-1].strip(),'device':'CPU','threads':1,'clock':'not locked'}}
    (ROOT/'VERIFICATION.json').write_text(json.dumps(ver,indent=2)+'\n')
    print('FINAL_VERDICT',json.dumps({'verdict':verdict,'f1_pairs':n_f1,'quality_all':nmse_ok},ensure_ascii=False),flush=True)


def main():
    pa=argparse.ArgumentParser();pa.add_argument('--phase',choices=['train','audit'],required=True)
    args=pa.parse_args()
    if args.phase=='train':run_train()
    else:run_audit()


if __name__=='__main__':main()
