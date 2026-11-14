#!/usr/bin/env python3
"""Fetch per-molecule rMD17 tensor files from the CC0-derived public mirror."""
import argparse,hashlib,json,pathlib,requests
REPO='StructureCloud/rMD17'; MOLECULES=['benzene','toluene','naphthalene','uracil']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[]
 for name in MOLECULES:
  p=out/f'{name}.pt';url=f'https://huggingface.co/datasets/{REPO}/resolve/main/preprocessed/{name}.pt'
  if not p.exists():
   with requests.get(url,stream=True,timeout=(30,180)) as r:
    r.raise_for_status()
    with p.open('wb') as f:
     for c in r.iter_content(1024*1024):
      if c:f.write(c)
  digest=hashlib.sha256(p.read_bytes()).hexdigest();rows.append({'molecule':name,'url':url,'path':str(p),'bytes':p.stat().st_size,'sha256':digest})
  print(name,p.stat().st_size,digest,flush=True)
 (out/'manifest.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
