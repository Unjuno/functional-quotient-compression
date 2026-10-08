"""Fetch the frozen MA-990 subject split from the public SONICOM WebDAV."""
import argparse,csv,hashlib,urllib.request
from pathlib import Path

BASE='https://transfer.ic.ac.uk:9090/2022_SONICOM-HRTF-DATASET'
SUBJECTS=tuple(range(1,43))

def fetch(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);rows=[]
    for subject in SUBJECTS:
        name=f'P{subject:04}_FreeFieldCompMinPhase_48kHz.sofa'
        path=root/name
        if not path.exists():
            url=f'{BASE}/P{subject:04}/HRTF/HRTF/48kHz/{name}'
            with urllib.request.urlopen(url,timeout=120) as response,path.open('wb') as out:
                while True:
                    block=response.read(1024*1024)
                    if not block:break
                    out.write(block)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append((subject,name,path.stat().st_size,digest))
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--provenance',default='data_provenance.csv');a=ap.parse_args()
    rows=fetch(a.output)
    with open(a.provenance,'w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['subject_id','file_name','bytes','sha256']);w.writerows(rows)
