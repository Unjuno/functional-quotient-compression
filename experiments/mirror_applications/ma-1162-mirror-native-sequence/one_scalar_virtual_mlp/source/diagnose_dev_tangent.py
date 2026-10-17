"""Post hoc *development-only* tangent interference diagnostic, not fresh tuning."""
import json
import torch
from torch.nn import functional as F
from source.model import VirtualMLP
from source.run import dataset, make_dense_from_mirror_start

torch.set_num_threads(1)


def generated_flat(m):
    return torch.cat([z.reshape(-1) for w,b in m.virtual_weights() for z in (w,b)])


def study(task,world=618320,seed=0):
    model=VirtualMLP('mirror',init_s=.2)
    eps=1e-6
    with torch.no_grad():
        model.s.add_(eps)
        plus=generated_flat(model)
        model.s.add_(-2*eps)
        minus=generated_flat(model)
        model.s.add_(eps)
    j=(plus-minus)/(2*eps)
    d=make_dense_from_mirror_start()
    x,y=dataset(task,world,seed,'train',n=512)
    F.mse_loss(d(x),y).backward()
    g=torch.cat([p.grad.reshape(-1) for p in d.parameters()])
    jac_norm=j.norm().item()
    grad_norm=g.norm().item()
    cos=float((j@g)/(j.norm()*g.norm()))
    return {'task':task,'teacher_world':world,'seed':seed,'scalar_model_ambient_dim':len(j),
            'tangent_rank_upper_bound':1,'tangent_vector_norm':jac_norm,
            'full_gradient_norm':grad_norm,'gradient_alignment_cosine':cos,
            'ambient_gradient_energy_reachable_locally':cos**2,
            'meaning':'Fraction of initial ambient parameter gradient aligned with the one-dimensional scalar manifold tangent; not a global approximation bound.'}

if __name__=='__main__':
    r=[study(t) for t in ('aligned','off_orbit')]
    print(json.dumps(r,indent=2))
    with open('results/DEVELOPMENT_TANGENT_DIAGNOSTIC.json','w') as f:json.dump(r,f,indent=2)
