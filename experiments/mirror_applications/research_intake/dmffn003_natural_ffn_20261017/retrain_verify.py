from pathlib import Path
import csv,json,hashlib
import torch
from experiment import split_world,fit_teacher,fit_student
from direct_labels import train as direct_train
from full_supervision import train as full_train
ROOT=Path(__file__).parent

def recheck():
 torch.set_num_threads(1);out=[]
 data=split_world(79021)
 teacher,_=fit_teacher(79021,data,None,updates=800)
 teacher_store=torch.load(ROOT/'models/teacher_world79021.pt',weights_only=True)
 assert all(torch.equal(val,teacher_store['state'][key]) for key,val in teacher.state_dict().items())
 out.append({'type':'teacher','checked_tensors':len(teacher.state_dict()),'tensor_exact':True})
 a_lr=json.loads((ROOT/'results/DEV_SELECTION.json').read_text())['selected_lr']
 ar=fit_student(79021,0,'mirror',teacher,data,a_lr,1800,(300,900,1800))
 out.append({'type':'distillation_mirror','same_sha':ar[-1]['model_sha256']==hashlib.sha256((ROOT/'models/79021_0_mirror.pt').read_bytes()).hexdigest()})
 d_lr=json.loads((ROOT/'results/DIRECT_DEV.json').read_text())['selected_lr']
 r=direct_train(79121,0,'mirror',split_world(79121),d_lr,2400,(300,900,2400))
 out.append({'type':'direct_mirror','same_sha':r[-1]['model_sha256']==hashlib.sha256((ROOT/'models/direct_79121_0_mirror.pt').read_bytes()).hexdigest()})
 f_lr=json.loads((ROOT/'results/FULL_DEV.json').read_text())['selected_lr']
 r=full_train(79221,0,'gain',split_world(79221),f_lr,2400,(300,900,2400))
 out.append({'type':'full_gain','same_sha':r[-1]['model_sha256']==hashlib.sha256((ROOT/'models/full_79221_0_gain.pt').read_bytes()).hexdigest()})
 r=full_train(79321,0,'mirror',split_world(79321),f_lr,6000,(900,2400,6000))
 out.append({'type':'long_mirror','same_sha':r[-1]['model_sha256']==hashlib.sha256((ROOT/'models/long_79321_0_mirror.pt').read_bytes()).hexdigest()})
 assert all(x.get('same_sha',True) for x in out)
 (ROOT/'results/RETRAIN.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':recheck()
