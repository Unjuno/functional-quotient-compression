"""Serialize selected MA-1059 development states as deployment payloads."""
import json,hashlib,sys
from pathlib import Path
import numpy as np,torch
import ma1059 as m

def main():
    root=m.RUNS/'dev'; sel=json.loads((root/'selection.json').read_text()); ids=m.DEV_IDS
    _,series=m.load_series(ids); pipe=m.load_pipe(); shared=torch.load(root/'shared.pt',map_location='cpu',weights_only=False); pooled=torch.load(root/'pooled.pt',map_location='cpu',weights_only=False)
    snap=m.SNAP; common=['model.safetensors','config.json','generation_config.json']; common_bytes=sum((snap/x).stat().st_size for x in common)
    out=root/'payloads'; out.mkdir(parents=True,exist_ok=True); manifest={'common_model_bytes':common_bytes,'methods':{},'series_ids':ids}
    def write(method,basestate,codes,resids=None):
        d=out/method;d.mkdir(parents=True,exist_ok=True); files=[]
        meta={'method':method,'series_ids':ids,'residual_ids':list(resids or [])}; p=d/'method.json';p.write_text(json.dumps(meta,indent=2)+'\n');files.append(p)
        if basestate is not None:
            for k in ('A','B'):
                p=d/f'{k}.f32le';p.write_bytes(basestate[k].numpy().astype('<f4').tobytes());files.append(p)
        if method=='native': arr=np.array([],dtype='<f4')
        elif method=='independent': arr=np.stack([np.concatenate([codes[s][0].numpy().ravel(),codes[s][1].numpy().ravel()]) for s in ids]).astype('<f4')
        elif method=='pooled_lora': arr=np.asarray(codes,dtype='<f4').reshape(-1)
        else: arr=np.stack([codes[s].numpy().ravel() for s in ids]).astype('<f4')
        p=d/'codes.f32le';p.write_bytes(arr.tobytes());files.append(p)
        if resids is not None and len(resids):
            p=d/'trend.f32le';p.write_bytes(np.asarray([resids[s] for s in ids if s in resids],dtype='<f4').tobytes());files.append(p)
        n=sum(p.stat().st_size for p in files);manifest['methods'][method]={'files':{p.name:p.stat().st_size for p in files},'adaptation_state_bytes':n,'complete_payload_bytes':common_bytes+n}
    write('native',None,np.array([],dtype='<f4'))
    write('pooled_lora',{'A':pooled['A'],'B':pooled['B']},pooled['fit_codes']['shared'].numpy())
    adapters={}
    for mode in ('mirror','diag','full','independent'):
        adapters[mode]={}
        for sid in ids:
            code,_=m.adapt(pipe,shared['A'],shared['B'],m.windows(series[sid])['support'],mode,int(sel['selected_updates'][mode]),105910+int(sel['selected_updates'][mode])+int(sid[1:]))
            adapters[mode][sid]=code
        write(mode,None if mode=='independent' else shared,adapters[mode])
    residuals={};rh=m.OutputLoRA(pipe,shared['A'],shared['B'],False); threshold=sel['residual_min_relative_gain']
    for sid in ids:
        w=m.windows(series[sid]); selected_resid=False
        if threshold is not None:
            cv,_=m.adapt(pipe,shared['A'],shared['B'],w['support'][:3],'mirror',int(sel['selected_updates']['mirror']),106300+int(sid[1:]))
            rc=m.trend_residual(pipe,rh,'mirror',cv,w['support'][:3],106400+int(sid[1:])); vctx,vy,_=w['support'][3]
            bp=m.predict(pipe,rh,'mirror',cv,vctx,106500+int(sid[1:])); den=m.mase_den(series[sid]); base=m.mase(bp,vy,den); corrected=m.mase(m.apply_resid(bp,rc),vy,den); gain=(base-corrected)/max(base,1e-8)
            selected_resid=gain>=float(threshold)
        if selected_resid: residuals[sid]=m.trend_residual(pipe,rh,'mirror',adapters['mirror'][sid],w['support'],106000+int(sid[1:]))
    rh.close()
    write('mirror_trend',shared,adapters['mirror'],residuals)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
