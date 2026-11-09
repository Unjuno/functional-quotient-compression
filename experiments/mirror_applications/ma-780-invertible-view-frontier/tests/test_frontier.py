import csv
from pathlib import Path
ROOT=Path(__file__).parents[1]

def test_source_rows_are_kept_unpooled():
    rows=list(csv.DictReader((ROOT/'RESULTS_CORE.csv').open(newline='')))
    assert len(rows)==13
    assert {r['source_ma'] for r in rows}=={'MA-771','MA-772','MA-775'}
    assert all(r['paired_comparable']=='false' for r in rows)
    for ma in ('MA-771','MA-772','MA-775'):
        subset=[r for r in rows if r['source_ma']==ma]
        assert any(r['method'].startswith('mirror_') for r in subset)
        assert any('exact payload and function alias' in r['status_note'] for r in subset)

def test_scope_forbids_cross_family_pooled_ranking():
    rows=list(csv.DictReader((ROOT/'FRONTIER_SCOPE.csv').open(newline='')))
    assert rows and all(r['pooled_rank_allowed']=='false' for r in rows)
