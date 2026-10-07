"""Read-only SRM003 audit. Exact payload, metric, leakage and routing checks."""
from pathlib import Path
import hashlib,json,subprocess,torch
from core import decode_model,encode_model
from engine import metrics

def main():
    torch.set_num_threads(1);root=Path('.')
    protocol=json.loads((root/'FROZEN_PROTOCOL.json').read_text())
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in protocol['source_hashes'].items()}
    assert all(hashes.values())
    model_rows=[];metric_rows=[];splits=[];routing=[]
    for folder in sorted((root/'results').glob('*')):
        if not (folder/'world.pt').exists():continue
        w=torch.load(folder/'world.pt',weights_only=True)
        group=[set(map(tuple,w[s+'_pairs'].tolist())) for s in ('train','dev','audit')]
        assert not group[0]&group[1] and not group[0]&group[2] and not group[1]&group[2]
        assert all(all((b,a) in z for a,b in z) for z in group)
        # Check realized random private operators are non-affine over GF(2)^4.
        affine=[]
        xx=torch.arange(16)[:,None];yy=torch.arange(16)[None,:]
        for t in w['table']:
            affine.append(bool((t[xx^yy]^t[0]==(t[xx]^t[yy])).all()))
        assert all(not affine[i] for i in range(24) if w['private_truth'][i])
        splits.append({'folder':str(folder),'pair_counts':[len(z) for z in group],'disjoint':True,'reverse_closed':True,'realized_private_nonaffine':True})
        for file in sorted(folder.rglob('*.bin')):
            raw=file.read_bytes();m=decode_model(raw);assert encode_model(m)==raw
            model_rows.append({'path':str(file),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'roundtrip':True})
        res=json.loads((folder/'result.json').read_text())
        for row in res['results']:
            m=decode_model((folder/row['kind']/'step1600.bin').read_bytes())
            which='audit' if 'audit_acc' in row['metrics'] else 'dev'
            ev=metrics(m,w,which);assert ev==row['metrics'],(folder,row['kind'],ev,row['metrics'])
            metric_rows.append({'folder':str(folder),'kind':row['kind'],'exact':True})
        if folder.name.startswith('fresh'):
            for kind in ('moe','lora','shared','hybrid'):
                m=decode_model((folder/kind/'step1600.bin').read_bytes())
                for bank_name in ('shared','private'):
                    bank=getattr(m.residual,bank_name)
                    if bank is None:continue
                    cache=[];handle=bank.router.register_forward_hook(lambda mod,inp,out:cache.append(out.detach()))
                    counts=torch.zeros(bank.n,dtype=torch.long)
                    with torch.no_grad():
                        for i in range(0,len(w['audit_targets']),256):
                            t=w['audit_tokens'][i:i+256];m(t)
                            scores=cache.pop().reshape(len(t),5,bank.n)[:,4]
                            ids=(scores.abs() if bank.signed else scores).topk(2,-1).indices
                            counts+=torch.bincount(ids.reshape(-1),minlength=bank.n)
                    handle.remove()
                    routing.append({'world':res['world'],'kind':kind,'bank':bank_name,'query_selection_counts':counts.tolist(),'used_slots':int((counts>0).sum()),'warning':'occupancy is not rule identity or quality'})
        if (folder/'long_result.json').exists():
            for row in json.loads((folder/'long_result.json').read_text()):
                m=decode_model((folder/(row['kind']+'_long')/'step4800.bin').read_bytes())
                assert metrics(m,w)==row['metrics']
                metric_rows.append({'folder':str(folder),'kind':row['kind']+'_long','exact':True})
    # Reconstructed saved models are checked independently of their measurement times.
    done={'frozen_training_source_unchanged':hashes,'payloads_audited':len(model_rows),'final_metric_rows_audited':len(metric_rows),
          'max_metric_difference':0.0,'split_audits':splits,'model_manifest':model_rows,'metric_rows':metric_rows,'routing_diagnostics':routing}
    Path('AUDIT.json').write_text(json.dumps(done,indent=2));print('AUDIT',len(model_rows),'payloads',len(metric_rows),'final metrics')
if __name__=='__main__':main()
