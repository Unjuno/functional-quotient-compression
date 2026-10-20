"""Post-training, read-only forensic replay and deployment packet checks.

The benchmark source/model.py and the frozen training protocol remain unmodified.
"""
from __future__ import annotations
import io
import json
from pathlib import Path
import torch
from model import initialize_model
from data import corpus

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'PROTOCOL.json').read_text())


def load_compact_model(packet:dict):
    """Rebuild executable model from only serialized inference payload fields.

    Requires *paid* source decoder/fixed program, which packet records as seed
    but whose implementation/program bytes are not included in torch.save.
    """
    blob=packet['graph_program'];variant=blob['decoder_kind']
    m=initialize_model(0,variant)
    # only direct decoder base tensors, no full training graph state
    base={k:v for k,v in packet.items() if k!='graph_program'}
    curr=m.state_dict()
    for k,v in base.items():curr[k]=v
    m.load_state_dict(curr,strict=True)
    with torch.no_grad():
        g=m.graph
        g.core.copy_(blob['global_core'])
        g.address.zero_();g.logits.zero_();g.active.fill_(False)
        ids=blob['edge_indices'].long()
        g.address[ids]=blob['addresses'];g.logits[ids]=blob['edge_gates']
        g.active[ids]=True
        g.node_bias.copy_(blob['node_bias']);g.layer_norm.weight.copy_(blob['norm_weight'])
        g.layer_norm.bias.copy_(blob['norm_bias'])
    return m


def validate():
    errors=[];rec=[]
    for world in CFG['synthetic_causal_language_data']['worlds']:
        for seed in CFG['synthetic_causal_language_data']['seeds']:
            data=corpus(world,'dev',seed,16)[:,:-1]
            for method in CFG['models']:
                ckpt=torch.load(ROOT/'checkpoints'/f'{world}_{seed}_{method}.pt',weights_only=False,map_location='cpu')
                a=initialize_model(seed,ckpt['variant']);a.load_state_dict(ckpt['model'])
                if not hasattr(a,'graph'):continue
                pkt={k:v.detach().cpu() for k,v in a.state_dict().items() if not k.startswith('graph.')}
                pkt['graph_program']=a.graph.deployed_edge_state()
                buf=io.BytesIO();torch.save(pkt,buf)
                if len(buf.getvalue())!=a.deploy_bytes():errors.append('bytes mismatch '+method)
                rebuilt=load_compact_model(torch.load(io.BytesIO(buf.getvalue()),weights_only=False))
                with torch.inference_mode():
                    prediction_a=a(data);prediction_b=rebuilt(data)
                    maxerr=float((prediction_a-prediction_b).abs().max())
                    original_active=a.graph.active.clone()
                    selected=[]
                    if a.graph.active_count()>=2:
                        selected=torch.nonzero(a.graph.active).flatten()[:2]
                        a.graph.active[selected]=False
                        cut_logit=a(data)
                        a.graph.active.copy_(original_active)
                        effect=float((prediction_a[:,-1]-cut_logit[:,-1]).abs().mean())
                    else:effect=None
                if maxerr>1e-6:errors.append('packet mismatch '+method)
                rec.append({'world':world,'seed':seed,'method':method,'max_logit_packet_error':maxerr,
                            'active_edges':a.graph.active_count(),
                            'causal_two_edge_cut_mean_abs_last_logit_change':effect,
                            'saved_inference_bytes':len(buf.getvalue())})
    result={'checks':len(rec),'errors':errors,'reconstruction_pass':not errors,
            'records':rec,'warning':'serialized inference packet excludes full Python execution program; deterministic buffers regenerated from source and seed'}
    (ROOT/'results'/'DEPLOY_REPLAY.json').write_text(json.dumps(result,indent=2)+'\n')
    print('deployment and causal intervention replay',len(rec),'errors',errors)
    return len(rec),errors

if __name__=='__main__':validate()
