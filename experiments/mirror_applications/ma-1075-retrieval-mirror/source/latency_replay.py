"""Interleaved CPU latency replay on identical development contexts (no fresh split)."""
import importlib.util, json, random, time, torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('screen',ROOT/'source'/'run_cpu_screen.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.set_num_threads(1)
train,dev,fresh,gm,genres,n_items,sha,item_map=m.data()
models={}
for kind in ['native','pair_gain','givens']:
    ck=torch.load(ROOT/'source'/f'{kind}.pt',map_location='cpu',weights_only=False)
    net=m.make_hstu(n_items); net.load_state_dict(ck['state_dict']); net.eval()
    head=None
    if kind!='native': head=m.RoleHead(len(genres),kind); head.load_state_dict(ck['head']); head.eval()
    vec=net.get_item_embeddings(torch.arange(1,n_items,dtype=torch.long))
    models[kind]=(net,head,vec)
rows=dev[:64]
with torch.no_grad():
    for row in rows:
        ids,lens,_,roles=m.batch([row],'cpu',gm)
        for kind in models:
            net,head,vec=models[kind]; q=m.encode(net,ids,lens)
            if head: q=head.transform(q,roles)
            _=q@vec.T
samples={k:[] for k in models}; rng=random.Random(1075)
with torch.no_grad():
    for it in range(768):
        row=rows[it%len(rows)]; ids,lens,_,roles=m.batch([row],'cpu',gm)
        order=list(models); rng.shuffle(order)
        for kind in order:
            net,head,vec=models[kind]; t=time.perf_counter(); q=m.encode(net,ids,lens)
            if head: q=head.transform(q,roles)
            _=q@vec.T; samples[kind].append(time.perf_counter()-t)
out={}
for k,v in samples.items():
    t=torch.tensor(v); out[k]={'n':len(v),'p50_s':float(t.quantile(.5)),'p95_s':float(t.quantile(.95)),'p99_s':float(t.quantile(.99)),'mean_s':sum(v)/len(v),'qps_at_p99':1/float(t.quantile(.99))}
(ROOT/'source'/'dev_latency_replay.json').write_text(json.dumps({'seed':1075,'torch_threads':1,'contexts':64,'samples_per_method':768,'data_sha256':sha,'results':out},indent=2)+'\n')
print(json.dumps(out,indent=2))
