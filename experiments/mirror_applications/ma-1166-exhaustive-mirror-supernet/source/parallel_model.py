"""MA-1166 Phase E — batched virtual decoder and full-edge synchronous graph.

Every module inherits the same *one* 253-scalar nn.Parameter from frozen Phase C
VirtualSupernet.  No auxiliary trainable node/edge weight parameters exist.
"""
import torch
from phase_c_model import VirtualSupernet


class ParallelGraph(VirtualSupernet):
    def __init__(self, decoder='mirror', mode='sync', seed=1977, steps=2):
        super().__init__(decoder=decoder,seed=seed)
        if mode not in ('serial_batched','sync'):
            raise ValueError(mode)
        if steps<1: raise ValueError('steps must be >= 1')
        self.mode=mode
        self.steps=int(steps)
        self.register_buffer('edge_src',torch.tensor([i for i,j,k in self.spec.edges],dtype=torch.long),persistent=False)
        self.register_buffer('edge_dst',torch.tensor([j for i,j,k in self.spec.edges],dtype=torch.long),persistent=False)
        self.register_buffer('all_indices',torch.arange(63,dtype=torch.long),persistent=False)
        self.register_buffer('slot_scale',torch.tensor([1.2/(3*j*self.steps) for j in range(1,7)],dtype=torch.float32),persistent=False)

    def _one_parallel_wave(self, states, weights):
        # states: [B,7,6] with slot0 fixed input; weights [63,110].
        batch=states.shape[0]
        source=states.index_select(1,self.edge_src).permute(1,0,2).contiguous()
        out=self._edge_result(source,weights)  # [63,B,6] one MLP batch
        gate=torch.sigmoid(self.theta[64:127]).reshape(63,1,1)
        contrib=gate*out
        # Gradient through index_add is a sum of the derivatives through all edges.
        agg=torch.zeros((7,batch,6),dtype=states.dtype,device=states.device)
        agg=agg.index_add(0,self.edge_dst,contrib)
        update=agg[1:].permute(1,0,2)*self.slot_scale.view(1,6,1)
        return torch.cat([states[:,:1,:],states[:,1:,:]+update],dim=1)

    def forward(self,x,return_stats=False):
        w=self.decode(self.all_indices)
        if self.mode=='serial_batched':
            states=[x]
            for j in range(1,7):
                idx=self.spec.dst[j]
                ins=torch.stack([states[self.spec.edges[q][0]] for q in idx.tolist()],dim=0)
                contrib=self._edge_result(ins,w[idx])*torch.sigmoid(self.theta[64:127][idx]).reshape(-1,1,1)
                states.append(states[-1]+contrib.sum(0)*(1.2/(3*j)))
            result=states[-1]
            active=63
        else:
            states=x.unsqueeze(1).expand(-1,7,-1).clone()
            for _ in range(self.steps):
                states=self._one_parallel_wave(states,w)
            result=states[:,6,:]
            active=63*self.steps
        if return_stats:
            return result,dict(candidate_edges=63,active_edge_evals=active,wave_count=self.steps if self.mode=='sync' else 6,physical_learned_scalars=253)
        return result

    def reference_sync(self,x):
        """Slow independent reference, deliberately uses Python loops, one edge at a time.
        Ground truth audit that the vectorized one-wave operator is correct.
        """
        assert self.mode=='sync'
        s=[x for _ in range(7)]
        for _ in range(self.steps):
            nxt=[x]
            for j in range(1,7):
                total=torch.zeros_like(x)
                for q in self.spec.dst[j].tolist():
                    src,_,_=self.spec.edges[q]
                    w=self.decode(torch.tensor([q],device=x.device))
                    out=self._edge_result(s[src].unsqueeze(0),w)[0]
                    total=total+torch.sigmoid(self.theta[64+q])*out
                nxt.append(s[j]+total*(1.2/(3*j*self.steps)))
            s=nxt
        return s[6]
