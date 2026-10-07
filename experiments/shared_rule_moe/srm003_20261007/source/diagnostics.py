"""Post-primary diagnostics specified before fresh evaluation; no selection on audit."""
import argparse, hashlib, json, time
from pathlib import Path
import torch
from core import decode_model, encode_model
from engine import stream, train, metrics

@torch.no_grad()
def predicted_intermediate_replay(m,w):
    m.eval();tok=w['audit_tokens'];target=w['audit_targets'];all_pred=[];valids=[]
    for i in range(0,len(target),256):
        t=tok[i:i+256];first=t.clone();first[:,3]=2;first[:,4]=0
        mid=m(first).argmax(-1)
        valid=(mid>=3)&(mid<19)
        second=t.clone();second[:,1]=mid;second[:,2]=t[:,3];second[:,3]=2;second[:,4]=0
        # Invalid first outputs are counted wrong; no truth is substituted.
        result=m(second).argmax(-1)
        result[~valid]=-1
        all_pred.append(result);valids.append(valid)
    pred=torch.cat(all_pred)
    return {'correct':int((pred==target).sum()),'n':len(target),'accuracy':float((pred==target).float().mean()),
            'valid_intermediate':int(torch.cat(valids).sum()),'forward_calls_per_example':2,
            'warning':'External composition control using same trained model twice; not single-forward or recurrent training performance.'}


def extend(root,world,seed):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    root=Path(root);w=torch.load(root/'world.pt',weights_only=True)
    x,y=stream(w,4800,seed+30000)
    torch.save({'tokens':x,'targets':y},root/'long_stream.pt')
    rows=[]
    for kind in ('dense','moe','hybrid'):
        m=decode_model((root/kind/'step1600.bin').read_bytes())
        before=metrics(m,w)
        replay_before=predicted_intermediate_replay(m,w)
        log=train(m,w,x,y,0,4800,.001,root/(kind+'_long'),checkpoints=(1600,4800),evaluate='audit')
        rows.append({'kind':kind,'world':world,'global_steps':6400,'before':before,'metrics':metrics(m,w),
                     'replay_before':replay_before,'replay_after':predicted_intermediate_replay(m,w),
                     'training':log,'bytes':len(encode_model(m))})
    (root/'long_result.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--world',type=int);p.add_argument('--seed',type=int)
    a=p.parse_args();extend(a.root,a.world,a.seed)
