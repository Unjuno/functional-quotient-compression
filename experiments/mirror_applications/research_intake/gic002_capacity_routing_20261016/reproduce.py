"""Reproduce in a NEW directory, preserving supplied historical evidence."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def lock(root,name,files):
    obj={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reproduction':True,'files':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}}
    (root/name).write_text(json.dumps(obj,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve()
    if out.exists():raise SystemExit('Refusing existing output directory: '+str(out))
    out.mkdir(parents=True)
    for pat in ['*.py','*.md','*PROTOCOL.json','requirements.txt']:
        for f in HERE.glob(pat):shutil.copy2(f,out/f.name)
    shutil.copytree(HERE/'tests',out/'tests',ignore=shutil.ignore_patterns('__pycache__'))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def run(*args):subprocess.run([sys.executable,*args],cwd=out,env=env,check=True)
    run('-m','pytest','-q')
    lock(out,'PRE_RUN_LOCK.json',['engine.py','runner.py','PROTOCOL.json','tests/test_engine.py'])
    run('runner.py','calibration','--workers','2');run('runner.py','fresh','--workers','2');run('analyze.py')
    lock(out,'WIDEN_LOCK.json',['widen.py','WIDEN_PROTOCOL.json','engine.py','FRESH_LOCK.json'])
    run('widen.py');run('analyze_widen.py');run('widen_references.py');run('diagnostics.py');run('audit.py','verify');run('audit.py','retrain');run('audit.py','benchmark')
if __name__=='__main__':main()
