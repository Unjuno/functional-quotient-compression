"""CPU screen using Meta's public HSTU implementation (GENREC_ROOT required)."""
from __future__ import annotations
import csv, hashlib, io, json, math, os, random, time, urllib.request, zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'source' / 'data'
URL = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
SEED = 41075
DIM = 32
MAX_SEQ = 50
BATCH = 64
TRAIN_CAP = 8000
EPOCHS = 2
NEGATIVES = 32


def register_reference_jagged_ops():
    # Reference-layout substitutes for FBGEMM's three jagged layout operations.
    # These are pure indexing/layout operations and keep autograd through copy slices.
    def cumsum(lengths):
        return torch.cat((torch.zeros(1, dtype=lengths.dtype, device=lengths.device), lengths.cumsum(0)))
    def jagged_to_padded(values, offsets, max_lengths, padding_value=0.0):
        off, n = offsets[0], max_lengths[0]
        shape = (off.numel() - 1, n, *values.shape[1:])
        result = values.new_full(shape, padding_value)
        for b in range(off.numel() - 1):
            a, z = int(off[b]), int(off[b + 1])
            if z > a:
                result[b, :min(z-a, n)] = values[a:min(z, a+n)]
        return result
    def dense_to_jagged(dense, offsets):
        off = offsets[0]
        return (torch.cat([dense[b, :int(off[b+1]-off[b])] for b in range(off.numel()-1)], dim=0),)
    torch.ops.fbgemm.asynchronous_complete_cumsum = cumsum
    torch.ops.fbgemm.jagged_to_padded_dense = jagged_to_padded
    torch.ops.fbgemm.dense_to_jagged = dense_to_jagged


def data():
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / 'ml-1m.zip'
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    raw = archive.read_bytes()
    (DATA / 'archive.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n')
    zf = zipfile.ZipFile(io.BytesIO(raw))
    ratings = []
    for line in zf.read('ml-1m/ratings.dat').decode('latin1').splitlines():
        u, i, rating, ts = map(int, line.split('::'))
        ratings.append((u, i, rating, ts))
    genres_by_source = {}
    all_genres = set()
    for line in zf.read('ml-1m/movies.dat').decode('latin1').splitlines():
        i, _, genres = line.split('::')
        gs = genres.split('|')
        genres_by_source[int(i)] = gs
        all_genres.update(gs)
    genre_names = sorted(all_genres)
    genre_id = {g:i for i,g in enumerate(genre_names)}
    item_ids = sorted({i for _,i,_,_ in ratings})
    item_map = {x:i+1 for i,x in enumerate(item_ids)}
    genres = np.zeros((len(item_ids)+1,len(genre_names)),dtype=np.float32)
    for src, idx in item_map.items():
        for g in genres_by_source.get(src,[]): genres[idx,genre_id[g]]=1
    users = {}
    for u,i,r,t in ratings: users.setdefault(u,[]).append((t,item_map[i],r))
    train, dev, fresh = [], [], []
    for u, seq in sorted(users.items()):
        seq.sort()
        if len(seq)<10: continue
        a=int(len(seq)*.8); b=int(len(seq)*.9)
        while a < len(seq) and seq[a-1][0] == seq[a][0]: a += 1
        b=max(b,a+1)
        while b < len(seq) and seq[b-1][0] == seq[b][0]: b += 1
        if a <= 2 or b <= a or b >= len(seq): continue
        for j in range(2,a):
            train.append((u, [x[1] for x in seq[max(0,j-MAX_SEQ):j]], seq[j][1], seq[j][0]))
        for j in range(a,b):
            dev.append((u, [x[1] for x in seq[max(0,j-MAX_SEQ):j]], seq[j][1], seq[j][0]))
        for j in range(b,len(seq)):
            fresh.append((u, [x[1] for x in seq[max(0,j-MAX_SEQ):j]], seq[j][1], seq[j][0]))
    return train,dev,fresh,genres,genre_names,len(item_ids)+1,hashlib.sha256(raw).hexdigest(),item_map


class IdentityPre(nn.Module):
    def forward(self,past_lengths,past_ids,past_embeddings,past_payloads):
        mask=(past_ids!=0).unsqueeze(-1).to(past_embeddings.dtype)
        return past_lengths,past_embeddings*mask,mask
class L2Post(nn.Module):
    def forward(self,x): return torch.nn.functional.normalize(x,dim=-1)


def make_hstu(num_items):
    root=os.environ.get('GENREC_ROOT','/tmp/genrec-ma1075')
    import sys
    sys.path.insert(0,root)
    register_reference_jagged_ops()
    from generative_recommenders.research.modeling.sequential.hstu import HSTU
    from generative_recommenders.research.modeling.sequential.embedding_modules import LocalEmbeddingModule
    from generative_recommenders.research.modeling.sequential.input_features_preprocessors import LearnablePositionalEmbeddingInputFeaturesPreprocessor
    from generative_recommenders.research.modeling.sequential.output_postprocessors import L2NormEmbeddingPostprocessor
    emb=LocalEmbeddingModule(num_items-1,DIM)
    pre=LearnablePositionalEmbeddingInputFeaturesPreprocessor(MAX_SEQ,DIM,0.0)
    post=L2NormEmbeddingPostprocessor(DIM)
    return HSTU(max_sequence_len=MAX_SEQ,max_output_len=1,embedding_dim=DIM,num_blocks=2,num_heads=1,linear_dim=DIM,attention_dim=DIM,normalization='rel_bias',linear_config='uvqk',linear_activation='silu',linear_dropout_rate=0.0,attn_dropout_rate=0.0,embedding_module=emb,similarity_module=None,input_features_preproc_module=pre,output_postproc_module=post,verbose=False)


class RoleHead(nn.Module):
    def __init__(self, n_genres, kind):
        super().__init__(); self.kind=kind
        # Codes are stored in FP32 and charged in the serialized model.
        self.code=nn.Parameter(torch.zeros(n_genres,DIM//2))
    def transform(self,u,roles):
        c=self.code[roles]
        if self.kind=='givens':
            x=u.reshape(-1,DIM//2,2); cs=torch.cos(c); sn=torch.sin(c)
            return torch.stack((x[...,0]*cs-x[...,1]*sn,x[...,0]*sn+x[...,1]*cs),dim=-1).reshape(-1,DIM)
        # Byte-matched ordinary per-pair gain control.
        gain=1.0+c
        return (u.reshape(-1,DIM//2,2)*gain.unsqueeze(-1)).reshape(-1,DIM)


def batch(rows,device,genre_matrix):
    b=len(rows); ids=torch.zeros((b,MAX_SEQ),dtype=torch.long,device=device); lengths=[]; tgt=[]; roles=[]
    for k,(_,ctx,item,_) in enumerate(rows):
        ctx=ctx[-MAX_SEQ:]; ids[k,:len(ctx)]=torch.tensor(ctx,device=device); lengths.append(len(ctx)); tgt.append(item)
        g=np.flatnonzero(genre_matrix[item]).tolist(); roles.append(g[0] if g else 0)
    return ids,torch.tensor(lengths,device=device),torch.tensor(tgt,device=device),torch.tensor(roles,device=device)


def encode(model, ids, lengths):
    embs=model.get_item_embeddings(ids)
    return model.encode(lengths,ids,embs,{},return_cache_states=False)


def evaluate(model,head,kind,rows,genre_matrix,n_items):
    model.eval(); by_role={}; t0=time.perf_counter()
    all_item_ids=torch.arange(1,n_items,dtype=torch.long)
    all_vectors=model.get_item_embeddings(all_item_ids)
    role_items={g:np.flatnonzero(genre_matrix[1:,g])+1 for g in range(genre_matrix.shape[1])}
    with torch.no_grad():
        for st in range(0,len(rows),32):
            chunk=rows[st:st+32]; ids,lens,targets,roles=batch(chunk,'cpu',genre_matrix); q=encode(model,ids,lens)
            for j,row in enumerate(chunk):
                item=int(targets[j]); gs=np.flatnonzero(genre_matrix[item]).tolist()
                for g in gs:
                    pool=role_items[g]
                    history=set(row[1]); pool=pool[~np.isin(pool, list(history))]
                    if item not in pool or len(pool)<2: continue
                    uq=q[j:j+1]
                    if head is not None: uq=head.transform(uq,torch.tensor([g]))
                    vec=all_vectors[torch.as_tensor(pool,dtype=torch.long)-1]
                    score=(uq*vec).sum(-1)
                    rank=int((score>score[np.where(pool==item)[0][0]]).sum())+1
                    gain=1/math.log2(rank+1) if rank<=10 else 0.0
                    by_role.setdefault(g,[]).append(gain)
    elapsed=time.perf_counter()-t0
    values={str(g):float(np.mean(v)) for g,v in by_role.items() if v}
    avg=float(np.mean(list(values.values()))) if values else 0.0
    # Isolated batch-1 CPU P99 for HSTU encode; candidate scoring excluded separately.
    sample=rows[:min(64,len(rows))]; times=[]
    with torch.no_grad():
        for row in sample:
            ids,lens,_,roles=batch([row],'cpu',genre_matrix); _=encode(model,ids,lens)
        for i in range(512):
            row=sample[i%len(sample)]; ids,lens,_,roles=batch([row],'cpu',genre_matrix); a=time.perf_counter(); q=encode(model,ids,lens)
            if head is not None: q=head.transform(q,roles)
            _=q @ all_vectors.T; times.append(time.perf_counter()-a)
    p99=float(np.quantile(times,.99)) if times else 0.0
    ids,lens,_,_=batch([sample[0]],'cpu',genre_matrix); _,cache=model.encode(lens,ids,model.get_item_embeddings(ids),{},return_cache_states=True)
    cache_bytes=sum(t.numel()*t.element_size() for layer in cache for t in layer)+len(sample[0][1])*8
    return {'mean_ndcg10':avg,'per_role':values,'eval_seconds':elapsed,'cpu_batch1_p99_s':p99,'cpu_batch1_qps':1/p99 if p99 else 0,'history_cache_bytes_including_item_ids':cache_bytes}


def run():
    torch.set_num_threads(max(1,min(4,os.cpu_count() or 1))); random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    train,dev,fresh,genre_matrix,genre_names,n_items,data_sha,item_map=data()
    # Development only: cap to a deterministic subset; fresh examples remain unopened by evaluation.
    train=sorted(train,key=lambda r:(hashlib.sha256(f'{SEED}:{r[0]}:{r[3]}'.encode()).digest()))[:TRAIN_CAP]
    rows=[]
    for kind in ['native','pair_gain','givens']:
        torch.manual_seed(SEED)
        model=make_hstu(n_items)
        head=RoleHead(len(genre_names),kind) if kind!='native' else None
        params=list(model.parameters())+(list(head.parameters()) if head else [])
        opt=torch.optim.AdamW(params,lr=1e-3,weight_decay=0.0)
        start=time.perf_counter(); model.train()
        for epoch in range(EPOCHS):
            order=list(range(len(train))); random.Random(SEED+epoch).shuffle(order)
            for st in range(0,len(order),BATCH):
                chunk=[train[x] for x in order[st:st+BATCH]]
                ids,lens,pos,roles=batch(chunk,'cpu',genre_matrix); u=encode(model,ids,lens)
                if head is not None: u=head.transform(u,roles)
                # Train next-item retrieval with the target item's first stable genre as the
                # candidate role; the role coordinate changes only the late query view.
                neg=torch.randint(1,n_items,(len(chunk),NEGATIVES)); cand=torch.cat((pos[:,None],neg),1)
                v=model.get_item_embeddings(cand)
                q=u[:,None,:].expand_as(v)
                logits=(q*v).sum(-1)
                loss=torch.nn.functional.cross_entropy(logits/0.05,torch.zeros(len(chunk),dtype=torch.long))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step()
        elapsed=time.perf_counter()-start
        metrics=evaluate(model,head,kind,dev,genre_matrix,n_items)
        payload=ROOT/'source'/f'{kind}.pt'; torch.save({'state_dict':model.state_dict(),'head':head.state_dict() if head else None,'genres':genre_names,'item_genres':genre_matrix,'item_id_map':item_map,'method':kind,'config':{'dim':DIM,'max_seq':MAX_SEQ,'blocks':2,'heads':1,'linear_dim':DIM,'attention_dim':DIM,'normalization':'rel_bias','linear_config':'uvqk','linear_activation':'silu'}},payload)
        rows.append({'condition':kind,'train_examples':len(train)*EPOCHS,'updates':math.ceil(len(train)/BATCH)*EPOCHS,'wall_s':elapsed,'loss_last':float(loss.detach()),'serialized_bytes':payload.stat().st_size,**metrics})
    out=ROOT/'source'/'development_seed41075.json'; out.write_text(json.dumps({'data_sha256':data_sha,'train_rows':len(train),'dev_rows':len(dev),'fresh_rows':len(fresh),'n_items':n_items,'genres':genre_names,'runs':rows},indent=2)+'\n')
    print(out.read_text())

if __name__=='__main__': run()
