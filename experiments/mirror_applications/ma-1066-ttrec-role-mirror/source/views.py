"""Pure view transforms used by MA-1066 and CPU unit tests."""
import torch

def apply_givens(x, angles):
    c=torch.cos(angles); s=torch.sin(angles); p=x.reshape(-1,16,2)
    return torch.stack((c*p[:,:,0]-s*p[:,:,1],s*p[:,:,0]+c*p[:,:,1]),dim=-1).reshape(-1,32)
