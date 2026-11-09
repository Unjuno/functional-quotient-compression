"""Validate and summarize selected invertible-view evidence without pooling."""
import csv
from pathlib import Path
ROOT=Path(__file__).parents[1]
rows=list(csv.DictReader((ROOT/'RESULTS_CORE.csv').open(newline='')))
assert len(rows)==13
assert all(r['paired_comparable']=='false' for r in rows)
for ma in ('MA-771','MA-772','MA-775'):
    subset=[r for r in rows if r['source_ma']==ma]
    assert subset
    aliases=[r for r in subset if 'exact payload and function alias' in r['status_note']]
    print(f'{ma}: {len(subset)} selected rows; {len(aliases)} exact Mirror/native alias rows; pooled ranking prohibited')
