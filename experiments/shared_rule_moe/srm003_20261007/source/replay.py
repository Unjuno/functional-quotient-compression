"""Re-execute two full main trainings and one selected-pruning continuation."""
from pathlib import Path
import hashlib,json,torch
from core import make_world,make_model,encode_model,decode_model
from engine import BASE,stream,train,prune_validation

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    root=Path('results/fresh66001');out=Path('results/replay66001');out.mkdir(exist_ok=True)
    c=json.loads(Path('CONFIGS.json').read_text())['configs'];w=make_world(66001);x,y=stream(w,1600,11701)
    parent=make_model(BASE,1701);train(parent,w,x,y,0,400,.003,out/'parent',evaluate='dev')
    rows=[]
    for kind in ('dense','hybrid'):
        m=make_model(c[kind],1711,parent)
        train(m,w,x,y,400,1200,.003,out/kind,checkpoints=(600,800,1200),evaluate='audit')
        train(m,w,x,y,1200,1600,.001,out/kind,evaluate='audit')
        same=(encode_model(m)==(root/kind/'step1600.bin').read_bytes())
        rows.append({'kind':kind,'steps':1600,'payload_exact':same});assert same
    p=decode_model((root/'hybrid/step1200.bin').read_bytes())
    m,sel=prune_validation(p,w,4);train(m,w,x,y,1200,1600,.001,out/'selected',evaluate='audit')
    same=encode_model(m)==(root/'selected/step1600.bin').read_bytes();assert same
    rows.append({'kind':'selected','steps_replayed':400,'selection_replayed':True,'payload_exact':same})
    Path('REPLAY.json').write_text(json.dumps(rows,indent=2));print('REPLAY PASS',rows)
if __name__=='__main__':main()
