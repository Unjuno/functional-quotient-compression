import csv, hashlib, json, random
from pathlib import Path
root = Path(__file__).resolve().parents[1]
record = json.loads((root / 'source' / 'random_draw.json').read_text())
pool_bytes = (root / 'source' / 'selection_pool.csv').read_bytes()
assert hashlib.sha256(pool_bytes).hexdigest() == record['pool_sha256']
rows = list(csv.DictReader(pool_bytes.decode().splitlines()))
assert len(rows) == record['pool_size']
index = random.Random(int(record['cryptographic_seed_hex'], 16)).randrange(len(rows))
assert index == record['uniform_index']
assert rows[index]['id'] == record['selected_id'] == 'MA-962'
print(json.dumps({'pool_hash':'PASS','uniform_draw_replay':'PASS','pool_size':len(rows),'index':index,'selected_id':rows[index]['id']}))
