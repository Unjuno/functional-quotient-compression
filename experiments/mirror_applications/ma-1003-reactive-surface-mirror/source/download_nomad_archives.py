#!/usr/bin/env python3
"""Fetch compact processed NOMAD archives for selected non-audit entries."""
import argparse, hashlib, json, pathlib, requests, time
BASE='https://nomad-lab.eu/prod/v1/api/v1/entries'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True);ap.add_argument('--roles',default='basis_fit,hyperparameter_development,few_shot_support');a=ap.parse_args()
 m=json.load(open(a.manifest));out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);wanted=set(a.roles.split(','));rows=[]
 for role,stages in m['roles'].items():
  if role not in wanted:continue
  for stage,facets in stages.items():
   for facet,groups in facets.items():
    for g in groups:
     for e in g['entries']:
      path=out/role/stage/facet/g['run_id']/(e['entry_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True)
      if not path.exists():
       payload={'required':{'run':{'system[-1]':'*','calculation[-1]':{'energy':'*','forces':'*','calculation_converged':'*'}}}}
       for attempt in range(8):
        r=requests.post(f"{BASE}/{e['entry_id']}/archive/query",json=payload,timeout=120)
        if r.status_code == 429 or r.status_code >= 500:
         time.sleep(float(r.headers.get('Retry-After', min(60, 2 ** attempt))))
         continue
        r.raise_for_status();path.write_bytes(r.content);time.sleep(0.25);break
       else: raise RuntimeError(f"NOMAD API retries exhausted for {e['entry_id']}")
      raw=path.read_bytes();rows.append({'role':role,'stage':stage,'facet':facet,'run_id':g['run_id'],'entry_id':e['entry_id'],'mainfile':e['mainfile'],'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
 man=out/'download_manifest.json';man.write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n');print('entries',len(rows),'bytes',sum(x['bytes'] for x in rows),'manifest_sha256',hashlib.sha256(man.read_bytes()).hexdigest())
if __name__=='__main__':main()
