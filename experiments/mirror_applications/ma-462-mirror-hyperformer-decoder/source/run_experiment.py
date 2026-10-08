"""MA-462 fixed-embedding generated-adapter decoder comparison."""
from __future__ import annotations
import argparse,hashlib,io,json,math,time
from pathlib import Path
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'source'
N_A=N_B=4;N_TASKS=N_A*N_B;INPUT=16;HIDDEN=32;OUTPUT=8;K=8;TRUE_RANK=4
ALPHAS=[0.0,0.25,0.5,0.75,1.0]
HOLDOUT_TASKS={0,6,11,13}
TRAIN_TASKS=[i for i in range(N_TASKS) if i not in HOLDOUT_TASKS]
HOLDOUT_LIST=sorted(HOLDOUT_TASKS)

class BaseNet(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Linear(INPUT,HIDDEN);self.head=nn.Linear(HIDDEN,OUTPUT)
    def features(self,x): return torch.tanh(self.encoder(x))
    def forward(self,x):
        h=self.features(x);return self.head(h),h


def task_attrs():return [(i//N_B,i%N_B) for i in range(N_TASKS)]

_WORLD_CACHE={}
def make_world(seed:int):
    if seed in _WORLD_CACHE:return _WORLD_CACHE[seed]
    g=torch.Generator().manual_seed(seed)
    teacher=BaseNet()
    with torch.no_grad():
        tw1=torch.randn(HIDDEN,INPUT,generator=g)*0.45;tb1=torch.randn(HIDDEN,generator=g)*0.1
        tw2=torch.randn(OUTPUT,HIDDEN,generator=g)*0.35;tb2=torch.randn(OUTPUT,generator=g)*0.05
        teacher.encoder.weight.copy_(tw1);teacher.encoder.bias.copy_(tb1);teacher.head.weight.copy_(tw2);teacher.head.bias.copy_(tb2)
    torch.manual_seed(seed+101);base=BaseNet();base.apply(lambda m: nn.init.xavier_uniform_(m.weight) if isinstance(m,nn.Linear) else None)
    base_opt=torch.optim.Adam(base.parameters(),lr=0.015)
    xpre=torch.randn(1024,INPUT,generator=torch.Generator().manual_seed(seed+202))
    with torch.no_grad():
        htrue=torch.tanh(xpre@tw1.T+tb1);ytrue=htrue@tw2.T+tb2
    t0=time.perf_counter()
    for _ in range(500):
        base_opt.zero_grad(set_to_none=True);yp=base(xpre)[0];loss=(yp-ytrue).square().mean();loss.backward();base_opt.step()
    pretrain_wall=time.perf_counter()-t0
    base.eval()
    for p in base.parameters():p.requires_grad_(False)
    emb_a=torch.randn(N_A,4,generator=g)*0.7;emb_b=torch.randn(N_B,4,generator=g)*0.7
    basis=torch.randn(K,HIDDEN,OUTPUT,generator=g)/math.sqrt(HIDDEN*K)
    pa=torch.randn(4,TRUE_RANK,generator=g)*0.8;pb=torch.randn(4,TRUE_RANK,generator=g)*0.8;dec=torch.randn(TRUE_RANK,K,generator=g)*0.8
    attrs=task_attrs();view=torch.stack([((emb_a[a]@pa)*(emb_b[b]@pb))@dec for a,b in attrs])
    view=view-view.mean(0,keepdim=True);view=view*(0.55/(view.square().mean().sqrt()+1e-8))
    private=torch.randn(N_TASKS,K,generator=g)*0.55
    deltas=torch.einsum('tk,kho->tho',view,basis)
    world={'seed':seed,'base':base,'emb_a':emb_a,'emb_b':emb_b,'basis':basis,'view_code':view,'private_code':private,
           'view_delta':deltas,'pretrain_wall_s':pretrain_wall,'pretrain_examples':1024*500,'pretrain_updates':500}
    _WORLD_CACHE[seed]=world;return world


def task_codes(world,alpha):
    return alpha*world['view_code']+(1-alpha)*world['private_code']


def data_for(world,alpha,seed,n_per_task):
    codes=task_codes(world,alpha);deltas=torch.einsum('tk,kho->tho',codes,world['basis'])
    g=torch.Generator().manual_seed(seed);xs=[];ys=[];tids=[];hs=[];bases=[]
    with torch.inference_mode():
        for tid in range(N_TASKS):
            x=torch.randn(n_per_task,INPUT,generator=g);base_y,h=world['base'](x);y=base_y+h@deltas[tid]
            xs.append(x);ys.append(y);hs.append(h);bases.append(base_y);tids += [tid]*n_per_task
    return {'x':torch.cat(xs),'y':torch.cat(ys),'h':torch.cat(hs),'base_y':torch.cat(bases),'task':torch.tensor(tids)}


class Decoder(nn.Module):
    METHOD_IDS={'zero':1,'linear':2,'hyper':3,'additive':4,'mirror':5,'mirror_private':6,'flat':7,'independent':8}
    def __init__(self,method,rank,world,init_seed):
        super().__init__();self.method=method;self.rank=rank;torch.manual_seed(init_seed)
        self.base=world['base']
        for p in self.base.parameters():p.requires_grad_(False)
        self.register_buffer('emb_a',world['emb_a'].clone());self.register_buffer('emb_b',world['emb_b'].clone())
        if method=='independent':self.register_buffer('adapter_basis',torch.empty(0))
        else:self.register_buffer('adapter_basis',world['basis'].clone())
        if method=='zero':pass
        elif method=='linear':self.linear=nn.Linear(8,K)
        elif method=='hyper':self.hyper=nn.Sequential(nn.Linear(8,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,K))
        elif method in ('additive','mirror','mirror_private'):
            self.proj_a=nn.Parameter(torch.randn(4,rank)*0.08);self.proj_b=nn.Parameter(torch.randn(4,rank)*0.08)
            self.decode=nn.Parameter(torch.randn(rank,K)*0.08);self.code_bias=nn.Parameter(torch.zeros(K))
            if method=='mirror_private':self.private=nn.Parameter(torch.zeros(N_TASKS,K))
        elif method=='flat':self.flat=nn.Parameter(torch.zeros(N_TASKS,K))
        elif method=='independent':
            self.full_delta=nn.Parameter(torch.zeros(N_TASKS,HIDDEN,OUTPUT))
        else:raise ValueError(method)
    def all_codes(self):
        ea=self.emb_a[torch.tensor([a for a,b in task_attrs()])]
        eb=self.emb_b[torch.tensor([b for a,b in task_attrs()])]
        if self.method=='zero':return torch.zeros(N_TASKS,K)
        if self.method=='linear':return self.linear(torch.cat([ea,eb],dim=-1))
        if self.method=='hyper':return self.hyper(torch.cat([ea,eb],dim=-1))
        if self.method in ('additive','mirror','mirror_private'):
            a=ea@self.proj_a;b=eb@self.proj_b
            z=a+b if self.method=='additive' else a*b
            code=z@self.decode+self.code_bias
            if self.method=='mirror_private':code=code+self.private
            return code
        if self.method=='flat':return self.flat
        return torch.zeros(N_TASKS,K)
    def task_deltas(self):
        if self.method=='independent':return self.full_delta
        return torch.einsum('tk,kho->tho',self.all_codes(),self.adapter_basis)
    def forward_from_features(self,h,base_y,task_ids,deltas=None):
        if deltas is None:deltas=self.task_deltas()
        d=deltas[task_ids]
        return base_y+torch.bmm(h.unsqueeze(1),d).squeeze(1)
    def trainable_parameters(self):return [p for p in self.parameters() if p.requires_grad]


def build_decoder(method,rank,world,seed):return Decoder(method,rank,world,seed)


def serialize(model,method,rank):
    meta={'format':'MA462-torch-v1','method':method,'rank':rank,'tasks':N_TASKS,'basis_atoms':K,'input_dim':INPUT,'hidden_dim':HIDDEN,'output_dim':OUTPUT,'conditioning':'fixed world-level factor embeddings'}
    bio=io.BytesIO();torch.save({'metadata':meta,'state_dict':model.state_dict()},bio);blob=bio.getvalue()
    return blob,hashlib.sha256(blob).hexdigest()


def decoder_gen_macs(method,rank):
    if method=='hyper':return 8*32+32*32+32*K
    if method=='linear':return 8*K
    if method in ('additive','mirror','mirror_private'):
        # Both factor maps cost 8*rank MACs, followed by the rank-by-K decode.
        # Elementwise product for Mirror is a multiply, not a MAC.
        return 8*rank+rank*K
    return 0


def train_one(method,rank,world,alpha,train_seed,updates=300):
    model=build_decoder(method,rank,world,world['seed']+METHOD_SEED[method]+rank*31)
    data=data_for(world,alpha,train_seed,32);keep=torch.isin(data['task'],torch.tensor(TRAIN_TASKS));h=data['h'][keep];base_y=data['base_y'][keep];y=data['y'][keep];tasks=data['task'][keep]
    params=model.trainable_parameters();opt=torch.optim.Adam(params,lr=0.012) if params else None
    start=time.perf_counter()
    if opt:
        for _ in range(updates):
            opt.zero_grad(set_to_none=True);pred=model.forward_from_features(h,base_y,tasks);loss=(pred-y).square().mean();loss.backward();opt.step()
    else:loss=(model.forward_from_features(h,base_y,tasks)-y).square().mean()
    wall=time.perf_counter()-start
    return model,wall,len(y)*updates,float(loss.detach())
METHOD_SEED={'zero':11,'linear':17,'hyper':23,'additive':29,'mirror':31,'mirror_private':37,'flat':41,'independent':43}


def evaluate(model,world,alpha,eval_seed):
    data=data_for(world,alpha,eval_seed,64);model.eval()
    t0=time.perf_counter()
    with torch.inference_mode():deltas=model.task_deltas()
    gen_wall=time.perf_counter()-t0
    held=torch.isin(data['task'],torch.tensor(HOLDOUT_LIST));seen=~held
    t0=time.perf_counter()
    with torch.inference_mode():pred=model.forward_from_features(data['h'],data['base_y'],data['task'],deltas)
    apply_wall=time.perf_counter()-t0
    err=(pred-data['y']).square().mean(dim=1);den=(data['y'].square().mean(dim=1)+1e-9)
    nerr=err/den
    held_metric=None if model.method in ('flat','independent') else float(nerr[held].mean())
    seen_metric=float(nerr[seen].mean())
    return {'seen_nmse':seen_metric,'heldout_nmse':held_metric,'decoder_generation_wall_s':gen_wall,
            'adapter_application_wall_s':apply_wall,'inference_wall_s':gen_wall+apply_wall,'eval_examples':len(data['x']),
            'transient_generated_adapter_bytes':N_TASKS*HIDDEN*OUTPUT*4,'adapter_gen_macs_per_task':decoder_gen_macs(model.method,model.rank),
            'adapter_apply_macs_per_example':HIDDEN*OUTPUT,'private_state_bytes':int(model.private.numel()*4) if model.method=='mirror_private' else 0}


def run_stage(stage,seeds,alphas,ranks,updates=300):
    rows=[]
    for seed in seeds:
        world=make_world(seed)
        for alpha in alphas:
            specs=[('zero',4),('linear',4),('hyper',4),('flat',4),('independent',4),('mirror_private',TRUE_RANK)]
            for r in ranks:specs += [('additive',r),('mirror',r)]
            for method,rank in specs:
                model,train_wall,exposures,train_loss=train_one(method,rank,world,alpha,seed+10000+int(alpha*1000),updates)
                metrics=evaluate(model,world,alpha,seed+20000+int(alpha*1000))
                blob,digest=serialize(model,method,rank)
                restored_obj=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True)
                restored=build_decoder(method,rank,world,world['seed']+METHOD_SEED[method]+rank*31);restored.load_state_dict(restored_obj['state_dict'])
                if not all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),restored.state_dict().values())):raise RuntimeError('payload roundtrip mismatch')
                if stage=='fresh':
                    folder=OUT/'payloads';folder.mkdir(exist_ok=True);name=f'fresh_{seed}_a{alpha:.2f}_{method}_r{rank}.pt';(folder/name).write_bytes(blob)
                task_count=len(TRAIN_TASKS)
                gen_macs=decoder_gen_macs(method,rank)
                apply_total=metrics['eval_examples']*HIDDEN*OUTPUT + N_TASKS*K*HIDDEN*OUTPUT
                train_proxy=updates*(len(TRAIN_TASKS)*gen_macs+len(TRAIN_TASKS)*K*HIDDEN*OUTPUT+len(TRAIN_TASKS)*32*HIDDEN*OUTPUT)*3
                rows.append({'world_seed':seed,'alpha':alpha,'method':method,'rank':rank,'updates':updates,'serialized_bytes':len(blob),'payload_sha256':digest,
                    'train_examples':exposures,'optimizer_updates':updates,'train_macs_proxy':train_proxy,'adapter_gen_macs_per_task':gen_macs,
                    'adapter_apply_macs_per_example':HIDDEN*OUTPUT,'inference_macs_proxy':apply_total,'pretrain_wall_s':world['pretrain_wall_s'],
                    'decoder_train_wall_s':train_wall,'train_loss':train_loss,**metrics,'stage':stage})
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['development','fresh'],required=True);args=ap.parse_args()
    if args.stage=='development':seeds=[4621,4622];alphas=ALPHAS;ranks=[2,4,8]
    else:
        frozen=json.loads((OUT/'frozen_config.json').read_text());seeds=frozen['fresh_seeds'];alphas=ALPHAS;ranks=[frozen['selected_rank']]
    rows=run_stage(args.stage,seeds,alphas,ranks,300)
    out=OUT/f'{args.stage}_raw.json';out.write_text(json.dumps({'stage':args.stage,'seeds':seeds,'alphas':alphas,'rows':rows},indent=2)+'\n')
    print(json.dumps({'stage':args.stage,'rows':len(rows),'output':str(out)},indent=2))
if __name__=='__main__':main()
