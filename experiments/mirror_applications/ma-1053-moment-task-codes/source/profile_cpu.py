#!/usr/bin/env python3
"""Single-task pre-head MOMENT CPU feasibility profile, before training."""
import json,time,torch
from pathlib import Path
from run_screen import load_moment
out=Path(__file__).resolve().parents[1]/'runs/cpu_profile.json'
load_start=time.perf_counter(); model=load_moment('forecasting'); load_s=time.perf_counter()-load_start; torch.set_num_threads(4)
x=torch.randn(1,1,512); mask=torch.ones(1,512)
t=time.perf_counter(); model.forecast(x_enc=x,input_mask=mask); tf=time.perf_counter()-t
x.requires_grad_(); t=time.perf_counter(); y=model.forecast(x_enc=x,input_mask=mask).forecast.square().mean(); y.backward(); tb=time.perf_counter()-t
obj={'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'threads':torch.get_num_threads(),'load_s':load_s,'synthetic_batch1_forward_s':tf,'synthetic_batch1_forward_backward_s':tb,'total_wall_s':time.perf_counter()-load_start,'model_revision':'5e44b0ea26376a176360f87831124e018f876d96'}
out.write_text(json.dumps(obj,indent=2)+'\n'); print(json.dumps(obj,indent=2))
