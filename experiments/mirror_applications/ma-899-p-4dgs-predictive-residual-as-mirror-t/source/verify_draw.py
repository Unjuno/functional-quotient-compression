import csv,hashlib,json,random
from pathlib import Path
root=Path(__file__).resolve().parents[1];rec=json.loads((root/'source/random_draw.json').read_text());raw=(root/'source/selection_pool.csv').read_bytes()
assert hashlib.sha256(raw).hexdigest()==rec['pool_sha256']
rows=list(csv.DictReader(raw.decode().splitlines()));assert len(rows)==rec['pool_size']
idx=random.Random(int(rec['cryptographic_seed_hex'],16)).randrange(len(rows))
assert idx==rec['uniform_index'] and rows[idx]['id']==rec['selected_id']=='MA-899'
print(json.dumps({'pool_hash':'PASS','uniform_draw_replay':'PASS','pool_size':len(rows),'index':idx,'selected_id':rows[idx]['id']}))
