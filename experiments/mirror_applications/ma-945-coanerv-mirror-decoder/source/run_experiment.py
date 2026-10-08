"""MA-945 fixed-protocol CPU video token-factor screen."""
from __future__ import annotations
import argparse,hashlib,io,json,math,resource,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import (FRAMES,HEIGHT,WIDTH,TOKENS,TOKEN_DIM,TIME_ROLES,REGION_ROLES,
                   BATCH_COORDS,CoordinateDecoder,SharedFactorGenerator,PrivateResidual,
                   all_coords,decode_y4m,evaluate,file_sha256,split_indices)
from model import N_HEADS

ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'source';DATA=Path('/tmp/ma945-data')
TRAIN_CLIPS=['akiyo_qcif','coastguard_qcif','foreman_qcif','news_qcif']
DEV_CLIPS=['carphone_qcif','container_qcif']
FRESH_CLIPS=['salesman_qcif','silent_qcif','grandma_qcif']
SEEDS=[9451,9452];FRESH_SEEDS=[94501,94502]
RANKS=[2,4,8];PRETRAIN_UPDATES=800;GENERATOR_UPDATES=600;CODE_UPDATES=600
METHODS=[('native_full',0),('lowrank',2),('lowrank',4),('lowrank',8),
         ('additive',2),('additive',4),('additive',8),('mirror',2),('mirror',4),('mirror',8),('mirror_private',4)]
METHOD_SEED={'native_full':1,'lowrank':7,'additive':13,'mirror':19,'mirror_private':23}


def load_clips(names):
    return {name:decode_y4m(DATA/f'{name}.y4m') for name in names}


def sample_batch(coords,targets,indices,generator,batch=BATCH_COORDS):
    take=indices[torch.randint(len(indices),(batch,),generator=generator)]
    return coords[take],targets[take]


def train_shared_decoder(train_videos,seed):
    torch.manual_seed(seed)
    decoder=CoordinateDecoder()
    tokens=nn.Parameter(torch.randn(len(train_videos),TOKENS,TOKEN_DIM)*.02)
    opt=torch.optim.Adam(list(decoder.parameters())+[tokens],lr=.001)
    samples={}
    for i,v in enumerate(train_videos):
        coords,targets=all_coords(v);g=torch.Generator().manual_seed(seed+100+i)
        idx=torch.arange(len(coords));samples[i]=(coords,targets,idx,g)
    g=torch.Generator().manual_seed(seed+200)
    start=time.perf_counter();loss_val=0.0
    decoder.train()
    for step in range(PRETRAIN_UPDATES):
        clip_i=step%len(train_videos);coords,targets,idx,vg=samples[clip_i]
        c,y=sample_batch(coords,targets,idx,vg)
        pred=decoder(c,tokens[clip_i]);loss=(pred-y).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();loss_val=float(loss.detach())
    wall=time.perf_counter()-start
    decoder.eval()
    for p in decoder.parameters():p.requires_grad_(False)
    banks=tokens.detach().cpu()
    return decoder,banks,wall,loss_val


def fit_shared_factor(train_banks,method,rank,seed):
    torch.manual_seed(seed)
    gen=SharedFactorGenerator(method,rank,len(train_banks))
    opt=torch.optim.Adam(gen.parameters(),lr=.001)
    target=train_banks
    start=time.perf_counter();last=0.0
    for _ in range(GENERATOR_UPDATES):
        pred=gen.train_tables();loss=(pred-target).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
    wall=time.perf_counter()-start
    with torch.no_grad():res=(target-gen.train_tables()).reshape(-1,TOKEN_DIM)
    _,_,vh=torch.linalg.svd(res,full_matrices=False)
    priv_basis=vh[:2].clone()
    return gen,priv_basis,wall,last


def fit_lowrank(train_banks,rank):
    matrix=train_banks.reshape(len(train_banks),-1)
    mean=matrix.mean(0);centered=matrix-mean
    _,_,vh=torch.linalg.svd(centered,full_matrices=False)
    eff=min(rank,vh.shape[0])
    basis=vh[:eff].reshape(eff,TOKENS,TOKEN_DIM).contiguous()
    return mean.reshape(TOKENS,TOKEN_DIM).contiguous(),basis


def copy_generator(shared,method,rank,n_train):
    gen=SharedFactorGenerator(method,rank,n_train)
    state={k:v for k,v in shared.state_dict().items() if k!='video_codes'}
    gen.load_state_dict(state,strict=False)
    for p in gen.parameters():p.requires_grad_(False)
    return gen


def adapt_video(decoder,video,clip_name,seed,method,rank,shared_factor=None,lowrank=None,priv_basis=None):
    torch.manual_seed(seed+METHOD_SEED[method]+rank)
    coords,targets=all_coords(video)
    train_idx,eval_idx=split_indices(len(coords),seed)
    rng=torch.Generator().manual_seed(seed+3000+METHOD_SEED[method]+rank)
    params=[];aux={};zdev=None
    if method=='native_full':
        token=nn.Parameter(torch.randn(TOKENS,TOKEN_DIM)*.02);params=[token]
        def make_tokens():return token
        aux['native_tokens']=token
    elif method=='lowrank':
        mean,basis=lowrank;eff=len(basis)
        zdev=nn.Parameter(torch.zeros(eff));params=[zdev]
        def make_tokens():return mean+torch.einsum('r,rnd->nd',zdev,basis)
        aux.update({'z':zdev,'mean':mean,'basis':basis,'effective_rank':eff})
    else:
        g=copy_generator(shared_factor, 'mirror' if method=='mirror_private' else method, rank, len(TRAIN_CLIPS))
        z0=shared_factor.video_codes.detach().mean(0).clone()
        zdev=nn.Parameter(z0);params=[zdev]
        if method=='mirror_private':
            pslot=nn.Parameter(torch.zeros(TOKENS,2));params.append(pslot)
            def make_tokens():return g.table_from_code(zdev)+pslot@priv_basis
            aux.update({'z':zdev,'private_slot':pslot,'private_basis':priv_basis,'generator':g})
        else:
            def make_tokens():return g.table_from_code(zdev)
            aux.update({'z':zdev,'generator':g})
    opt=torch.optim.Adam(params,lr=.001)
    decoder.eval();start=time.perf_counter();last=0.0
    for _ in range(CODE_UPDATES):
        c,y=sample_batch(coords,targets,train_idx,rng)
        pred=decoder(c,make_tokens());loss=(pred-y).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();last=float(loss.detach())
    train_wall=time.perf_counter()-start
    t0=time.perf_counter()
    with torch.no_grad():token_state=make_tokens().detach().cpu()
    codegen_wall=time.perf_counter()-t0
    eval_mask=torch.zeros(len(coords),dtype=torch.bool);eval_mask[eval_idx]=True
    metrics=evaluate(decoder,token_state,video,eval_mask)
    return {'clip':clip_name,'seed':seed,'method':method,'rank':rank,'tokens':token_state,
            'train_code':{k:v.detach().cpu() for k,v in aux.items() if isinstance(v,torch.Tensor)},
            'train_code_wall_s':train_wall,'token_generation_wall_s':codegen_wall,
            'train_loss':last,'train_examples':int(CODE_UPDATES*BATCH_COORDS),
            'optimizer_updates':CODE_UPDATES,'eval_mask_count':int(eval_mask.sum()),
            'heldout_split_indices_sha256':hashlib.sha256(eval_idx.numpy().tobytes()).hexdigest(),
            **metrics}


def decoder_macs_per_pixel():
    h=TOKEN_DIM;d=TOKEN_DIM//3;n=TOKENS
    qpe=20*h;q=h*h;k_v=2*n*h*h;attn=2*n*h;out=h*h;mlp=h*48+48*3
    return int(qpe+q+k_v/BATCH_COORDS+attn+out+mlp)


def generator_cost(method,rank):
    if method=='native_full':return {'generator_macs_per_token':0,'generator_elementwise_multiplies_per_token':0}
    if method=='lowrank':return {'generator_macs_per_token':rank*TOKEN_DIM,'generator_elementwise_multiplies_per_token':0}
    base=rank*TOKEN_DIM
    return {'generator_macs_per_token':base,'generator_elementwise_multiplies_per_token':(3*rank if method in ('mirror','mirror_private') else 0)}


def codegen_for(method,rank,record):
    if method=='native_full':return {'tokens':record['tokens']}
    if method=='lowrank':return {'z':record['train_code']['z']}
    d={'video_code':record['train_code']['z']}
    if method=='mirror_private':d['private_slot_code']=record['train_code']['private_slot']
    return d


def serialize_library(decoder,method,rank,records,shared_factor=None,lowrank=None,priv_basis=None):
    rep={}
    if method=='native_full':rep={'video_tokens':torch.stack([x['tokens'] for x in records])}
    elif method=='lowrank':
        mean,basis=lowrank;rep={'mean_token_table':mean,'shared_token_basis':basis,
            'video_codes':torch.stack([x['train_code']['z'] for x in records])}
    else:
        rep={'shared_factor_generator':{k:v for k,v in shared_factor.shared_state().items()},
             'video_codes':torch.stack([x['train_code']['z'] for x in records])}
        if method=='mirror_private':rep.update({'private_token_basis':priv_basis,
            'private_slot_codes':torch.stack([x['train_code']['private_slot'] for x in records])})
    metadata={'format':'MA945-inference-v1','method':method,'rank':rank,'video_ids':[x['clip'] for x in records],
              'frames':FRAMES,'height':HEIGHT,'width':WIDTH,'token_count':TOKENS,'token_dim':TOKEN_DIM,
              'token_role_order':'slot=4*time_role+spatial_quadrant','color':'RGB','shared_decoder':'reduced CoANeRV coordinate-query core'}
    obj={'metadata':metadata,'decoder_state':{k:v.detach().cpu() for k,v in decoder.state_dict().items()},'representation':rep}
    bio=io.BytesIO();torch.save(obj,bio);blob=bio.getvalue()
    return blob,hashlib.sha256(blob).hexdigest()


def marginal_bytes(method,rank,record,shared_factor=None,lowrank=None,priv_basis=None):
    obj={'method':method,'rank':rank,'video_id':record['clip'],'code':codegen_for(method,rank,record)}
    b=io.BytesIO();torch.save(obj,b);return len(b.getvalue())


def run_development(seeds=SEEDS,dev_names=DEV_CLIPS,updates_pre=PRETRAIN_UPDATES,updates_code=CODE_UPDATES):
    # These are fixed protocol constants; alternate update arguments are used only by quick unit tests.
    global PRETRAIN_UPDATES,CODE_UPDATES
    PRETRAIN_UPDATES=updates_pre;CODE_UPDATES=updates_code
    all_names=TRAIN_CLIPS+list(dev_names);videos=load_clips(all_names);rows=[]
    out_dir=SOURCE/'payloads';out_dir.mkdir(exist_ok=True)
    for seed in seeds:
        train_videos=[videos[n] for n in TRAIN_CLIPS]
        decoder,banks,pre_wall,pre_loss=train_shared_decoder(train_videos,seed)
        factor_cache={};low_cache={}
        for method,rank in METHODS:
            if method in ('additive','mirror','mirror_private'):
                base_method='mirror' if method=='mirror_private' else method
                key=(base_method,rank)
                if key not in factor_cache:
                    g,pb,wall,loss=fit_shared_factor(banks,base_method,rank,seed+100+rank+METHOD_SEED[base_method])
                    factor_cache[key]=(g,pb,wall,loss)
            elif method=='lowrank':
                if rank not in low_cache:low_cache[rank]=fit_lowrank(banks,rank)
        results=[]
        for ci,clip in enumerate(dev_names):
            for method,rank in METHODS:
                shared=priv=None;low=None;gen_wall=gen_loss=0.0
                if method in ('additive','mirror','mirror_private'):
                    base_method='mirror' if method=='mirror_private' else method
                    shared,priv,gen_wall,gen_loss=factor_cache[(base_method,rank)]
                elif method=='lowrank':low=low_cache[rank]
                rec=adapt_video(decoder,videos[clip],clip,seed+500+ci*100,method,rank,shared,low,priv)
                rec.update({'base_seed':seed,'decoder_pretrain_wall_s':pre_wall,'decoder_pretrain_loss':pre_loss,
                    'shared_generator_fit_wall_s':gen_wall,'shared_generator_fit_loss':gen_loss,
                    **generator_cost(method,rank),'decoder_macs_per_query_proxy':decoder_macs_per_pixel(),
                    'active_attention_logits_bytes':BATCH_COORDS*N_HEADS*TOKENS*4,
                    'peak_process_rss_kib':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)})
                if method in ('additive','mirror','mirror_private'):rec['_shared']=shared;rec['_priv_basis']=priv
                if method=='lowrank':rec['_lowrank']=low
                results.append(rec)
        # Emit full K=1 and K=2 payloads for every method/rank; these are the authoritative storage measurements.
        groups={}
        for rec in results:groups.setdefault((rec['method'],rec['rank']),[]).append(rec)
        for (method,rank),group in groups.items():
            shared=group[0].get('_shared');priv=group[0].get('_priv_basis');low=group[0].get('_lowrank')
            one_blobs=[]
            for rec in group:
                blob,digest=serialize_library(decoder,method,rank,[rec],shared,low,priv)
                one_blobs.append((len(blob),digest))
                rec['single_video_payload_bytes']=len(blob);rec['single_video_payload_sha256']=digest
                rec['marginal_video_code_bytes']=marginal_bytes(method,rank,rec,shared,low,priv)
            blob,digest=serialize_library(decoder,method,rank,group,shared,low,priv)
            path=out_dir/f'dev_seed{seed}_{method}_r{rank}_K{len(group)}.pt';path.write_bytes(blob)
            library_bytes=len(blob)
            for rec in group:
                rec.update({'library_k':len(group),'library_payload_bytes':library_bytes,'library_payload_sha256':digest,
                    'payload_artifact':str(path.relative_to(ROOT)),
                    'training_examples_shared_decoder':PRETRAIN_UPDATES*BATCH_COORDS,
                    'training_updates_shared_decoder':PRETRAIN_UPDATES,
                    'training_token_examples_generator':GENERATOR_UPDATES*len(TRAIN_CLIPS)*TOKENS if method in ('additive','mirror','mirror_private') else 0,
                    'training_updates_generator':GENERATOR_UPDATES if method in ('additive','mirror','mirror_private') else 0,
                    'training_examples_code_fit':rec['train_examples'],
                    'total_training_examples_including_shared':PRETRAIN_UPDATES*BATCH_COORDS+GENERATOR_UPDATES*len(TRAIN_CLIPS)*TOKENS+rec['train_examples'],
                    'total_optimizer_updates_including_shared':PRETRAIN_UPDATES+(GENERATOR_UPDATES if method in ('additive','mirror','mirror_private') else 0)+rec['optimizer_updates'],
                    'inference_macs_proxy':FRAMES*HEIGHT*WIDTH*decoder_macs_per_pixel()+TOKENS*rec['generator_macs_per_token'],
                    'training_macs_proxy':(PRETRAIN_UPDATES*BATCH_COORDS*decoder_macs_per_pixel()*3+
                       (GENERATOR_UPDATES*len(TRAIN_CLIPS)*TOKENS*rank*TOKEN_DIM*3 if method in ('additive','mirror','mirror_private') else 0)+
                       rec['optimizer_updates']*BATCH_COORDS*(decoder_macs_per_pixel()+TOKENS*rec['generator_macs_per_token'])*3)})
        def result_row(rec):
            return {k:v for k,v in rec.items() if not k.startswith('_') and k not in ('tokens','train_code')}
        rows.extend(result_row(x) for x in results)
        (SOURCE/'development_partial.json').write_text(json.dumps({'stage':'development','completed_seeds':[x for x in seeds if any(r['base_seed']==x for r in rows)],'rows':rows},indent=2)+'\n')
    return rows


def summarize(rows):
    by={}
    for r in rows:by.setdefault((r['method'],r['rank']),[]).append(r)
    gates=[]
    for rank in (2,4,8):
        for clip in DEV_CLIPS:
            for seed in SEEDS:
                m=next(x for x in rows if x['method']=='mirror' and x['rank']==rank and x['clip']==clip and x['base_seed']==seed)
                n=next(x for x in rows if x['method']=='native_full' and x['clip']==clip and x['base_seed']==seed)
                candidates=[x for x in rows if x['method'] in ('lowrank','additive') and x['clip']==clip and x['base_seed']==seed]
                simple=min(candidates,key=lambda x:x['heldout_psnr_db'])
                # Require a simple control within 10% library bytes for Mirror-specific comparison.
                matched=[x for x in candidates if abs(x['library_payload_bytes']-m['library_payload_bytes'])<=.10*m['library_payload_bytes']]
                best_matched=max(matched,key=lambda x:x['heldout_psnr_db']) if matched else None
                tests={'psnr_within_1db':m['heldout_psnr_db']>=n['heldout_psnr_db']-1.0,
                    'ssim_within_003':m['heldout_ssim']>=n['heldout_ssim']-.03,
                    'library_bytes_095':m['library_payload_bytes']<=.95*n['library_payload_bytes'],
                    'marginal_code_half':m['marginal_video_code_bytes']<=.5*n['marginal_video_code_bytes'],
                    'throughput_half':m['query_pixels_per_second']>=.5*n['query_pixels_per_second'],
                    'matched_simple_03db':bool(best_matched and m['heldout_psnr_db']>=best_matched['heldout_psnr_db']+.30)}
                gates.append({'rank':rank,'clip':clip,'seed':seed,'mirror_psnr_db':m['heldout_psnr_db'],'native_psnr_db':n['heldout_psnr_db'],
                    'mirror_ssim':m['heldout_ssim'],'native_ssim':n['heldout_ssim'],'mirror_library_bytes':m['library_payload_bytes'],
                    'native_library_bytes':n['library_payload_bytes'],'mirror_marginal_bytes':m['marginal_video_code_bytes'],
                    'native_marginal_bytes':n['marginal_video_code_bytes'],'matched_best_simple':None if best_matched is None else {'method':best_matched['method'],'rank':best_matched['rank'],'psnr':best_matched['heldout_psnr_db'],'bytes':best_matched['library_payload_bytes']},'tests':tests})
    rank_pass={str(rank):all(all(v for v in g['tests'].values()) for g in gates if g['rank']==rank) for rank in (2,4,8)}
    return {'experiment_id':'MA-945','stage':'development','rows':len(rows),'base_seeds':SEEDS,'dev_clips':DEV_CLIPS,
        'rank_pass':rank_pass,'gate_checks':gates,'fresh_accessed':False,
        'decision':'pending'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['development'],required=True);args=ap.parse_args()
    torch.set_num_threads(1)
    rows=run_development()
    (SOURCE/'development_raw.json').write_text(json.dumps({'stage':'development','rows':rows},indent=2)+'\n')
    (SOURCE/'development_partial.json').unlink(missing_ok=True)
    summary=summarize(rows)
    passing=[int(k) for k,v in summary['rank_pass'].items() if v]
    summary['decision']='PASS-development-gate' if passing else 'FAIL-development-gate; fresh clips remain unopened'
    summary['selected_rank']=min(passing) if passing else None
    summary['fresh_accessed']=False
    (SOURCE/'development_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'summary':summary['decision'],'rank_pass':summary['rank_pass']},indent=2))
if __name__=='__main__':main()
