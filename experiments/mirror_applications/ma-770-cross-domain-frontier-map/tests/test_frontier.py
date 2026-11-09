import csv,importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/'source'/'build_frontier.py'
s=importlib.util.spec_from_file_location('ma770',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_map_has_separate_domains_and_traceable_reports():
    m.main()
    rows=list(csv.DictReader((P.parents[1]/'FRONTIER_MAP.csv').open(newline='')))
    assert len(rows)==len(m.DATA)==16
    assert len({r['experiment_id'] for r in rows})==16
    assert all((m.ROOT/r['source_report']).is_file() for r in rows)

def test_no_pooled_cross_domain_metric():
    m.main()
    rows=list(csv.DictReader((P.parents[1]/'FRONTIER_MAP.csv').open(newline='')))
    assert not any('aggregate' in k.lower() or 'overall_score' in k.lower() for k in rows[0])

def test_byte_savings_recomputed():
    for r in m.DATA:
        cand,base,pct=int(r[2]),int(r[3]),float(r[4])
        assert abs((100*(1-cand/base))-pct)<.15
