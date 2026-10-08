import csv,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
d=json.loads((root/"source/random_draw.json").read_text())
raw=(root/"source/selection_pool.csv").read_bytes()
rows=list(csv.DictReader(raw.decode().splitlines()))
assert hashlib.sha256(raw).hexdigest()==d["pool_csv_sha256"]
seed=bytes.fromhex(d["seed_hex"]);n=len(rows);limit=(1<<256)//n*n;c=0
while True:
 value=int.from_bytes(hashlib.sha256(seed+c.to_bytes(4,"big")).digest(),"big")
 if value<limit:break
 c+=1
idx=value%n
assert n==d["eligible_pool_size"] and idx==d["index_zero_based"]==542
assert rows[idx]["id"]==d["selected_row"]["id"]=="MA-1138"
assert d["baseline_commit"]=="c935a903daca5c7d1d48aa50d05b5bd50f239cba"
assert "MA-1138" not in d["remote_ma_branch_ids_at_draw"]
print(json.dumps({"pool_sha256":"PASS","uniform_draw_replay":"PASS","selected_id":rows[idx]["id"],"pool_size":n},indent=2))
