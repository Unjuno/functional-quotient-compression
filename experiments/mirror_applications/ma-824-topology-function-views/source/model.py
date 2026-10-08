"""Exact two-node programmable graph substrate for MA-824."""
import hashlib
import torch
from torch.nn import functional as F

METHODS=('mirror_factors','native_interpreter','hard_shared_indices','independent_programs')
TOPOLOGIES=('chain','parallel');FUNCS=('tanh_mlp','sine_mlp')


def function_bank(seed):
    g=torch.Generator().manual_seed(seed+3400)
    return [{'w1':torch.randn(4,1,generator=g)*.9,'b1':torch.randn(4,generator=g)*.2,
             'w2':torch.randn(1,4,generator=g)*.7,'b2':torch.randn(1,generator=g)*.1} for _ in FUNCS]


def op(x,weights,fid):
    h=F.linear(x,weights['w1'],weights['b1'])
    h=torch.tanh(h) if fid==0 else torch.sin(h)
    return F.linear(h,weights['w2'],weights['b2']).squeeze(-1)


def execute(x,bank,topology,fid):
    if topology==0:return op(op(x[:,0:1],bank[fid],fid).unsqueeze(-1),bank[fid],fid)
    return op(x[:,0:1],bank[fid],fid)+op(x[:,1:2],bank[fid],fid)


def make_inputs(seed,split,n=8192):
    off={'development':1_000_000,'fresh':2_000_000}[split]
    g=torch.Generator().manual_seed(seed+off)
    return torch.randn(n,2,generator=g)*1.5


def payload_for(method,seed,bank):
    if method=='mirror_factors':
        body={'shared_function_bank':bank,'topology_codes':{'chain':0,'parallel':1},
              'function_codes':{'tanh_mlp':0,'sine_mlp':1},
              'decoder':'compose(topology_code,function_code); no per-program pair table'}
    elif method=='native_interpreter':
        body={'shared_function_bank':bank,'primitive_signatures':{'tanh_mlp':0,'sine_mlp':1},
              'graph_grammar':{'chain':[[0,1],[1,2]],'parallel':[[0,2],[1,2]]},
              'program_signatures':[['chain','tanh_mlp'],['chain','sine_mlp'],['parallel','tanh_mlp']],
              'heldout_rule':'compose registered graph grammar with an existing function signature'}
    elif method=='hard_shared_indices':
        body={'shared_function_bank':bank,'index_tables':{'topology':[0,1],'function':[0,1]},
              'decoder':'hard table lookup'}
    else:
        programs=[]
        for topology in TOPOLOGIES:
            for fid,name in enumerate(FUNCS):
                nodes=[{'function_name':name,'function_weights':bank[fid]} for _ in range(2)]
                programs.append({'topology':topology,'nodes':nodes,
                    'edges':[[0,1],[1,2]] if topology=='chain' else [[0,2],[1,2]]})
        body={'independent_programs':programs}
    return {'schema':'MA-824/program-inference-v1','seed':seed,'method':method,
        'input_dim':2,'node_hidden_dim':4,'topologies':TOPOLOGIES,'function_names':FUNCS,
        'heldout_pair':['parallel','sine_mlp'],'representation':body}


def save_payload(path,method,seed,bank):
    p=payload_for(method,seed,bank);torch.save(p,path);raw=path.read_bytes()
    return {'path':path.name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def execute_payload(x,payload,topology,fid):
    body=payload['representation']
    if payload['method']=='independent_programs':
        idx=TOPOLOGIES.index(topology)*2+fid
        bank=[body['independent_programs'][idx]['nodes'][0]['function_weights']]
        return execute(x,bank+[bank[0]],TOPOLOGIES.index(topology),fid)
    return execute(x,body['shared_function_bank'],TOPOLOGIES.index(topology),fid)
