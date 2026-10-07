"""Run the prespecified pair-only diagnostic serially (two models per world)."""
import json
from pathlib import Path
from engine import run_world
configs=json.loads(Path('CONFIGS.json').read_text())['configs']
for w,s in zip([66001,66002,66003],[1701,1702,1703]):
    run_world(Path('results')/f'paironly{w}',w,s,configs,.003,True,True,False,['dense','hybrid'])
