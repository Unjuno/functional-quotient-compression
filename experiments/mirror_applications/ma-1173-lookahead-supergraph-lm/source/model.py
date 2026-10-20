"""MA-1173: finite, causally masked Mirror supergraph for small autoregressive LMs.

No claim of arbitrary MLP expressivity or independent virtual parameters.
"""
from __future__ import annotations
import io
import math
import torch
from torch import nn
from torch.nn import functional as F

D=12
VOCAB=24
SEQ_LEN=12
NODES=3
VIEWS=2
STEPS=2
EDGES=NODES*NODES*VIEWS
PAGES=4
HIDDEN=16


def fixed_decoder(seed: int = 1173):
    gen=torch.Generator().manual_seed(seed)
    n=2*D*D
    perm=[];signs=[]
    for k in range(PAGES):
        perm.append(torch.randperm(n, generator=gen))
        signs.append((torch.randint(0,2,(n,), generator=gen,dtype=torch.int64).float()*2-1))
    u=torch.randn(D,generator=gen);u=u/u.norm()
    house=torch.eye(D)-2*torch.outer(u,u)
    return torch.stack(perm),torch.stack(signs),house


class VirtualSupergraph(nn.Module):
    def __init__(self, transform:str="mirror"):
        super().__init__()
        assert transform in ("mirror","direct_tanh","affine")
        self.transform=transform
        self.core=nn.Parameter(torch.randn(2,D,D)*0.14)
        self.address=nn.Parameter(torch.randn(EDGES,PAGES)*0.04)
        self.logits=nn.Parameter(torch.zeros(EDGES))
        self.node_bias=nn.Parameter(torch.randn(NODES,D)*0.02)
        self.layer_norm=nn.LayerNorm(D)
        perm,signs,house=fixed_decoder()
        self.register_buffer('perms',perm,persistent=False)
        self.register_buffer('signs',signs,persistent=False)
        self.register_buffer('house',house,persistent=False)
        src,dst,view=[],[],[]
        for d in range(NODES):
            for s in range(NODES):
                for v in range(VIEWS):
                    src.append(s);dst.append(d);view.append(v)
        self.register_buffer('sources',torch.tensor(src),persistent=False)
        self.register_buffer('targets',torch.tensor(dst),persistent=False)
        self.register_buffer('views',torch.tensor(view),persistent=False)
        self.register_buffer('active',torch.ones(EDGES,dtype=torch.bool))

    def edge_weights(self, selected:torch.Tensor | None =None):
        if selected is None:
            selected=torch.arange(EDGES,device=self.core.device)
        if len(selected)==0:
            return self.core.new_zeros((0,2,D,D))
        raw=self.core.reshape(-1)
        pages=raw[self.perms]*self.signs  # [PAGES, 2*D*D]
        mix=F.softmax(self.address[selected]*2.0,dim=-1)
        weights=(mix@pages).reshape(-1,2,D,D)
        if self.transform=='mirror':
            h=self.house
            # Nonlinear input/output basis change, not merely a change of gauge.
            weights=h @ (torch.tanh(2.0*(h @ weights @ h))/2.0) @ h
        elif self.transform=='direct_tanh':
            weights=torch.tanh(2.0*weights)/2.0
        return weights

    def forward(self, x:torch.Tensor, forced_active:torch.Tensor | None=None):
        # x: [batch, seq, d]; all tokenwise edges remain independent across tokens
        active=self.active if forced_active is None else forced_active
        idx=torch.nonzero(active,as_tuple=False).flatten()
        state=x.unsqueeze(2)+self.node_bias
        if len(idx)==0:
            return state.mean(dim=2)
        weights=self.edge_weights(idx)
        gates=torch.sigmoid(self.logits[idx]); src=self.sources[idx]; dst=self.targets[idx]
        for _ in range(STEPS):
            z=state[:,:,src,:] # [B,L,E,D]
            h=F.gelu(torch.einsum('blei,eij->blej',z,weights[:,0]))
            y=torch.einsum('blei,eij->blej',h,weights[:,1])
            message=state.new_zeros(state.shape)
            message=message.index_add(2,dst,y*gates.view(1,1,-1,1))
            state=self.layer_norm(state+0.25*message/(NODES*VIEWS))
        return state.mean(dim=2)

    def cut(self,edge:int):
        if self.active[edge].item() is False: raise ValueError('edge already disabled')
        self.active[edge]=False

    def active_count(self):return int(self.active.sum().item())

    def deployed_edge_state(self):
        """Storage payload, not a custom bitstream/real kernel.

        The global core remains paid; only active address+gate rows are retained.
        An executable deployment also requires the decoder program, embedding, and attention.
        """
        ids=torch.nonzero(self.active).flatten()
        return {'global_core':self.core.detach().cpu().contiguous(),
                'addresses':self.address.detach().cpu()[ids].contiguous(),
                'edge_gates':self.logits.detach().cpu()[ids].contiguous(),
                'edge_indices':ids.cpu().to(torch.int16),
                'node_bias':self.node_bias.detach().cpu().contiguous(),
                'norm_weight':self.layer_norm.weight.detach().cpu().contiguous(),
                'norm_bias':self.layer_norm.bias.detach().cpu().contiguous(),
                'decoder_seed':1173,'decoder_kind':self.transform,'nodes':NODES,
                'views':VIEWS,'steps':STEPS}


class TinyCausalLM(nn.Module):
    def __init__(self,variant='mirror'):
        super().__init__()
        assert variant in ('mirror','direct_tanh','affine','native')
        self.variant=variant
        self.embed=nn.Embedding(VOCAB,D)
        self.pos=nn.Embedding(SEQ_LEN,D)
        self.embed_ln=nn.LayerNorm(D)
        self.attn_in=nn.Linear(D,3*D)
        self.attn_out=nn.Linear(D,D)
        self.attn_ln=nn.LayerNorm(D)
        if variant=='native':
            self.native=nn.Sequential(nn.Linear(D,HIDDEN),nn.GELU(),nn.Linear(HIDDEN,D))
            self.native_ln=nn.LayerNorm(D)
        else:
            self.graph=VirtualSupergraph(variant)
        self.output_ln=nn.LayerNorm(D)
        self.head=nn.Linear(D,VOCAB)

    def forward(self,tokens:torch.Tensor,forced_active:torch.Tensor|None=None):
        b,t=tokens.shape
        if t>SEQ_LEN:raise ValueError('sequence too long')
        h=self.embed_ln(self.embed(tokens)+self.pos(torch.arange(t,device=tokens.device)))
        q,k,v=self.attn_in(h).chunk(3,dim=-1)
        heads=2; dh=D//heads
        q=q.reshape(b,t,heads,dh).transpose(1,2)
        k=k.reshape(b,t,heads,dh).transpose(1,2)
        v=v.reshape(b,t,heads,dh).transpose(1,2)
        mix=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.)
        mix=mix.transpose(1,2).contiguous().reshape(b,t,D)
        h=self.attn_ln(h+self.attn_out(mix))
        if self.variant=='native':
            # Keep independent MLP at ~same parameter budget as the virtual graph.
            for _ in range(STEPS):h=self.native_ln(h+0.25*self.native(h))
        else:
            h=h+self.graph(h,forced_active=forced_active)
        return self.head(self.output_ln(h))

    def model_loss(self,x:torch.Tensor,y:torch.Tensor):
        logits=self(x)
        per=F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1),reduction='none').reshape(y.shape)
        # Count normal autoregressive NLL separately; query position is most diagnostic.
        return per.mean()+2.0*per[:,-1].mean()

    def deploy_bytes(self):
        if self.variant=='native':
            buffer=io.BytesIO();torch.save({k:v.detach().cpu() for k,v in self.state_dict().items()},buffer)
            return len(buffer.getvalue())
        packet={k:v.detach().cpu() for k,v in self.state_dict().items() if not k.startswith('graph.')}
        packet['graph_program']=self.graph.deployed_edge_state()
        buffer=io.BytesIO();torch.save(packet,buffer)
        return len(buffer.getvalue())


def initialize_model(seed:int,variant:str):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TinyCausalLM(variant)
