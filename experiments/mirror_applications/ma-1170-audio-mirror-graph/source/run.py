"""Frozen MA-1170 CPU screening. Separate train/audit so fresh is sealed until all fits finish.
Usage:
  python source/run.py train --output .
  python source/run.py audit --output .
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,os,platform,statistics,time
from pathlib import Path
import torch
import numpy as np
from model import AudioAE,program_sha256
from audio import main_data,cross_data

KINDS=['mirror','affine','sparse','native']
SEEDS=[31,37,41]


def hashfile(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def nmse(yhat,y):
    return float((yhat-y).square().sum().item()/(y.square().sum().item()+1e-16))

def scales_for(model,train):
    with torch.no_grad():
        codes=model.encode(train)
        return torch.quantile(codes,0.995,dim=0).clamp_min(0.01)/255.0

def quantize(z,scale):
    return torch.clamp(torch.round(z/scale),0,255).to(torch.uint8)

def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def train(out, protocol):
    norm,mu,sd,meta=main_data()
    rows=[]
    for seed in SEEDS:
        for kind in KINDS:
            torch.manual_seed(seed)
            model=AudioAE(kind)
            optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['training']['lr'],weight_decay=protocol['training']['weight_decay'])
            rng=torch.Generator().manual_seed(seed*731+5)
            best_dev=float('inf');best_weights=None;best_update=-1
            for step in range(1,501):
                ix=torch.randint(len(norm['train']),(96,),generator=rng)
                x=norm['train'][ix]
                optimizer.zero_grad(set_to_none=True)
                pred=model(x)
                loss=(pred-x).square().mean()
                if kind in ('mirror','affine'):
                    loss=loss+0.0005*torch.sigmoid(model.graph.gates).sum()
                assert torch.isfinite(loss)
                loss.backward()
                optimizer.step()
                if step%50==0:
                    model.eval()
                    with torch.no_grad(): dev=nmse(model(norm['dev']),norm['dev'])
                    model.train()
                    if dev<best_dev:
                        best_dev=dev; best_update=step
                        best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(best_weights)
            model.eval()
            scale=scales_for(model,norm['train'])
            with torch.no_grad():
                tr=nmse(model(norm['train']),norm['train']);dv=nmse(model(norm['dev']),norm['dev'])
                # Confirm train-stage has never used fresh/late tensors from norm.
            ckp={'id':'MA-1170','seed':seed,'kind':kind,'weights':best_weights,'dev_nmse':best_dev,'best_update':best_update,
                 'train_nmse':tr,'scale':scale,'norm_mean':mu,'norm_std':sd,'audio_sha256':meta['audio_sha256'],
                 'program_sha256':program_sha256(),'protocol_sha256':hashfile(out/'PROTOCOL.json')}
            chkpath=out/'checkpoints'/f'{kind}_{seed}.pt'
            torch.save(ckp,chkpath)
            row={'seed':seed,'kind':kind,'train_nmse':tr,'dev_nmse':dv,'best_update':best_update,'checkpoint_sha256':hashfile(chkpath)}
            rows.append(row)
            print('TRAIN',json.dumps(row),flush=True)
    write_csv(out/'results'/'TRAIN_DEV_ONLY.csv',rows)
    assert len(rows)==12
    manifest={'stage':'training complete; fresh sealed','checkpoints':len(rows),'protocol_sha256':hashfile(out/'PROTOCOL.json'),
              'frames':meta['frames'],'program_sha256':program_sha256(), 'dev_only_metrics':True}
    (out/'results'/'TRAIN_VERIFICATION.json').write_text(json.dumps(manifest,indent=2))
    print('TRAIN_COMPLETE',json.dumps(manifest),flush=True)


def timing(model,x, repeats=60):
    with torch.inference_mode():
        for _ in range(20):model(x)
        times=[]
        for _ in range(repeats):
            start=time.perf_counter_ns()
            model(x)
            times.append((time.perf_counter_ns()-start)/1000)
    return float(statistics.median(times)),float(np.percentile(times,10)),float(np.percentile(times,90))


def spectrum_ablation(model,x):
    with torch.inference_mode():
        z=model.encode(x)
        y0=model.decode(z)
        fractions=[]
        for j in range(16):
            if int((z[:,j]!=0).sum())<3:continue
            z2=z.clone();z2[:,j]=0
            effect=(model.decode(z2)-y0).abs().mean(0)
            frac=float(torch.topk(effect,k=8).values.sum()/(effect.sum()+1e-12))
            fractions.append(frac)
        return float(statistics.median(fractions)) if fractions else float('nan'),len(fractions)


def packet_bytes(kind,model,checkpoint):
    pack={'version':'MA-1170/v1','kind':kind,'state':model.state_dict(),
          'train_mu':checkpoint['norm_mean'],'train_sd':checkpoint['norm_std'],
          'quant_scales':checkpoint['scale'],'program_sha256':checkpoint['program_sha256'],
          'description':'torch.save stable inference payload; program deterministic seed=1170, code overhead separately'}
    b=io.BytesIO();torch.save(pack,b)
    raw=b.getvalue()
    rp=io.BytesIO(raw)
    recovered=torch.load(rp,weights_only=True)
    test=AudioAE(recovered['kind'])
    test.load_state_dict(recovered['state'])
    with torch.inference_mode():
        probe=torch.linspace(-0.5,0.5,steps=64).reshape(1,64)
        maxdiff=float((test(probe)-model(probe)).abs().max())
    return len(raw),maxdiff


def audio_stream_bytes(out,kind,seed,codes):
    # 12-byte header: 'MA17', number of frames uint32 little endian, number of dims uint32.
    import struct
    body=struct.pack('<4sII',b'MA17',codes.shape[0],codes.shape[1])+codes.contiguous().numpy().tobytes()
    assert len(body)==12+codes.shape[0]*codes.shape[1]
    path=out/'results'/f'fresh_u8_{kind}_{seed}.bin'
    path.write_bytes(body)
    return len(body)


def audit(out, protocol):
    all_files=[out/'checkpoints'/f'{k}_{s}.pt' for s in SEEDS for k in KINDS]
    if not all(p.exists() for p in all_files):
        raise RuntimeError('Fresh audit forbidden: training not complete')
    verpath=out/'results'/'TRAIN_VERIFICATION.json'
    if not verpath.exists():raise RuntimeError('Training verification absent')
    stage=json.loads(verpath.read_text())
    assert stage['checkpoints']==12
    assert stage['protocol_sha256']==hashfile(out/'PROTOCOL.json')
    norm,mu,sd,meta=main_data()
    cross=cross_data(mu,sd)
    results=[]
    for seed in SEEDS:
        for kind in KINDS:
            ck=torch.load(out/'checkpoints'/f'{kind}_{seed}.pt',weights_only=True)
            assert ck['seed']==seed and ck['kind']==kind and ck['protocol_sha256']==stage['protocol_sha256']
            assert ck['audio_sha256']==meta['audio_sha256'] and ck['program_sha256']==program_sha256()
            assert torch.equal(ck['norm_mean'],mu) and torch.equal(ck['norm_std'],sd)
            model=AudioAE(kind);model.load_state_dict(ck['weights']);model.eval()
            with torch.inference_mode():
                z=model.encode(norm['fresh']);qy=quantize(z,ck['scale']); yquant=model.decode(qy.float()*ck['scale'])
                fresh=nmse(model(norm['fresh']),norm['fresh'])
                quant=nmse(yquant,norm['fresh'])
                late=nmse(model(norm['late']),norm['late'])
                cross_loss=nmse(model(cross),cross)
                prune=(nmse(model(norm['fresh'],masked_edges=8),norm['fresh']) if kind in ('mirror','affine') else None)
                loc,nactive=spectrum_ablation(model,norm['fresh'])
                b1,p10_1,p90_1=timing(model,norm['fresh'][:1]);b64,p10_64,p90_64=timing(model,norm['fresh'][:64])
                pkt,packet_err=packet_bytes(kind,model,ck)
                stream=audio_stream_bytes(out,kind,seed,qy)
                effective=stream+pkt
                actual_feature_bytes=len(norm['fresh'])*64*4
                # No phase / waveform bits counted; this is feature rate only.
                result={
                    'seed':seed,'kind':kind,'parameters':sum(p.numel() for p in model.parameters()),'best_update':ck['best_update'],
                    'dev_nmse':ck['dev_nmse'],'fresh_nmse':fresh,'fresh_u8_nmse':quant,'late_nmse':late,'cross_nmse':cross_loss,
                    'top8_localization':loc,'active_latents':nactive,'mask8_nmse':prune,
                    'packet_bytes':pkt,'fresh_code_bytes':stream,'fresh_total_bytes':effective,'fresh_fp32_feature_bytes':actual_feature_bytes,
                    'cpu_b1_median_us':b1,'cpu_b1_p10_us':p10_1,'cpu_b1_p90_us':p90_1,
                    'cpu_b64_median_us':b64,'cpu_b64_p10_us':p10_64,'cpu_b64_p90_us':p90_64,
                    'packet_replay_max_abs':packet_err}
                results.append(result)
                print('AUDIT',json.dumps(result),flush=True)
    write_csv(out/'RESULTS_CORE.csv',results)
    groups={k:[r for r in results if r['kind']==k] for k in KINDS}
    paired=[r['fresh_nmse']/next(rr['fresh_nmse'] for rr in groups['affine'] if rr['seed']==r['seed']) for r in groups['mirror']]
    quality=all(r<=0.95 for r in paired) and statistics.median([r['fresh_nmse'] for r in groups['mirror']])<statistics.median([r['fresh_nmse'] for r in groups['sparse']])
    loc_pass=all(m['top8_localization']>=max(a['top8_localization'],s['top8_localization'])+0.05
                 and m['fresh_nmse']<=min(a['fresh_nmse'],s['fresh_nmse'])*1.05
                 for m,a,s in zip(groups['mirror'],groups['affine'],groups['sparse']))
    ver={'id':'MA-1170','protocol_sha256':hashfile(out/'PROTOCOL.json'),'source_sha256':meta['audio_sha256'],
         'program_sha256':program_sha256(),'n_runs':len(results),'checkpoint_replay':all(r['packet_replay_max_abs']==0.0 for r in results),
         'finite_all':all(np.isfinite(r['fresh_nmse']) and np.isfinite(r['fresh_u8_nmse']) for r in results),
         'frames':meta['frames'],'paired_mirror_affine_fresh_nmse_ratio':paired,
         'mirror_quality_pass':quality,'mirror_localization_pass':loc_pass,
         'model_status':'PASS' if quality and loc_pass else 'FAIL',
         'hardware':{'platform':platform.platform(),'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads(),
             'cpu':open('/proc/cpuinfo').read().split('model name\t: ',1)[-1].split('\n',1)[0],
             'clock':'not pinned, frequency scaling may affect wall time','gpu':False},
         'caveats':['single thirty-second file and 3 seeds','mel magnitude cannot reconstruct audio phase',
                    'feature code-rate not waveform quality and not lossless','No external codec reproduced',
                    'full-source inference code size not included in torch-save state packet',
                    '500-update comparison not capacity or convergence']}
    (out/'VERIFICATION.json').write_text(json.dumps(ver,ensure_ascii=False,indent=2,default=str)+'\n')
    print('AUDIT_COMPLETE',json.dumps(ver,ensure_ascii=False,default=str),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['train','audit'])
    p.add_argument('--output',type=Path,default=Path('.'))
    args=p.parse_args();out=args.output.resolve()
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    torch.set_num_threads(1)
    config=json.loads((out/'PROTOCOL.json').read_text())
    if args.phase=='train':train(out,config)
    else:audit(out,config)
