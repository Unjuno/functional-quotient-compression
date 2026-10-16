"""Representability oracle: reconstruct known aligned teacher without fitting.
Separates optimization gap from absent function family capacity. No test labels used to fit.
"""
import json,torch
from pathlib import Path
from model import DualSwiGLU,generate_world
ROOT=Path(__file__).resolve().parents[1]
def main():
    torch.set_num_threads(1)
    rows=[]
    for w in [73111,73112,73113,73114,73115,73211,73212,73213]:
        world=generate_world(w,'aligned')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(w)
            teacher=DualSwiGLU('mirror',d=8,h=24,K=3,q=12)
            with torch.no_grad():teacher.gangle.uniform_(-0.8,0.8);teacher.uangle.uniform_(-0.8,0.8)
        x,y=world['test']
        with torch.no_grad():
            z=teacher(x)
            mx=(z-y).abs().max().item();loss=((z-y)**2).mean().item()
        rows.append({'world':w,'teacher_exactly_in_mirror12_family':bool(mx==0), 'max_abs_diff':mx,'MSE':loss,'teacher_train_noise_only':True})
        assert mx == 0,('aligned family construction incorrect',w,mx)
    path=ROOT/'results/EXACT_ORACLE.json';path.write_text(json.dumps({'worlds':rows,'all_exact':True},indent=2)+'\n')
    print('ALL',len(rows),'teacher oracles perfectly representable, zero max delta')
if __name__=='__main__':main()
