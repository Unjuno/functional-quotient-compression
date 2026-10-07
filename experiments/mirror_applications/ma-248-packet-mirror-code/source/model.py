"""Tiny phase-slot decoder for MA-248."""
from __future__ import annotations
import io, json
import torch
from torch import nn

class SlotBlock(nn.Module):
    def __init__(self,d,heads):
        super().__init__();self.attn=nn.MultiheadAttention(d,heads,batch_first=True);self.n1=nn.LayerNorm(d);self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d));self.n2=nn.LayerNorm(d)
    def forward(self,x,mask):
        a,_=self.attn(x,x,x,attn_mask=mask,need_weights=False);x=self.n1(x+a);return self.n2(x+self.ff(x))

class PacketDecoder(nn.Module):
    METHODS=("ptp","direct","broadcast","scalar_gate","mirror","untied")
    def __init__(self,method,period,codes,states=16,rules=8,d=16):
        super().__init__()
        if method not in self.METHODS:raise ValueError(method)
        self.method,self.period,self.codes=method,period,codes;self.states,self.rules,self.d=states,rules,d
        self.state_emb=nn.Embedding(states,d);self.rule_emb=nn.Embedding(rules,d)
        self.context_pos=nn.Parameter(torch.randn(2,d)*.02);self.phase=nn.Parameter(torch.randn(period,d)*.02)
        if method=="ptp":self.ptp_bit=nn.Embedding(2,d)
        if method in ("broadcast","scalar_gate","mirror"):self.code_emb=nn.Embedding(codes,d)
        if method=="untied":self.phase_code=nn.Parameter(torch.randn(period,codes,d)*.02)
        if method=="scalar_gate":self.phase_gate=nn.Parameter(torch.ones(period))
        if method=="mirror":self.angles=nn.Parameter(torch.zeros(period,d//2))
        self.blocks=nn.ModuleList([SlotBlock(d,4) for _ in range(2)]);self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,states)
    def _phase_inputs(self,bits,code):
        b=bits.shape[0]
        if self.method=="ptp":z=self.ptp_bit(bits.long())
        elif self.method=="direct":z=torch.zeros((b,self.period,self.d),device=bits.device)
        elif self.method=="untied":
            z=self.phase_code[None].expand(b,-1,-1,-1).gather(2,code[:,None,None,None].expand(-1,self.period,1,self.d)).squeeze(2)
        else:
            z=self.code_emb(code)[:,None,:].expand(-1,self.period,-1)
            if self.method=="scalar_gate":z=z*self.phase_gate[None,:,None]
            elif self.method=="mirror":
                pairs=z.reshape(b,self.period,self.d//2,2);ang=self.angles[None];c,s=torch.cos(ang),torch.sin(ang);a,q=pairs[...,0],pairs[...,1]
                z=torch.stack((c*a-s*q,s*a+c*q),dim=-1).reshape(b,self.period,self.d)
        return z+self.phase[None]
    def forward(self,rule,start,bits,code):
        context=torch.stack((self.rule_emb(rule),self.state_emb(start)),1)+self.context_pos[None]
        x=torch.cat((context,self._phase_inputs(bits,code)),1);mask=torch.ones((2+self.period,2+self.period),dtype=torch.bool,device=x.device).triu(1)
        for block in self.blocks:x=block(x,mask)
        return self.head(self.norm(x[:,2:]))
    def parameter_count(self):return sum(p.numel() for p in self.parameters())
    def serialize(self):
        buf=io.BytesIO();cfg={"method":self.method,"period":self.period,"codes":self.codes,"states":self.states,"rules":self.rules,"d":self.d,"blocks":len(self.blocks)}
        torch.save({"state_dict":self.state_dict(),"config":cfg},buf)
        return buf.getvalue()+json.dumps(cfg,sort_keys=True).encode()
    def serialized_payload_bytes(self):return len(self.serialize())

def compute_proxy(period,d,examples,method):
    seq=period+2;base=4*seq*d*d+2*seq*d*(2*d)+seq*d*16+2*seq*seq*d
    extra=period*d if method=="ptp" else (period*d//2 if method=="mirror" else 0)
    return int((base+extra)*examples*3)
