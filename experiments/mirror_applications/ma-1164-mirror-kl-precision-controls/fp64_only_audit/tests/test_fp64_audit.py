import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from paired_fp64_audit import audit

def test_full_precision_matched_subset():
    data=pd.read_csv(ROOT/'results/fp64_subset.csv')
    summary, detail=audit(data)
    assert len(data)==108 and len(summary)==15 and len(detail)==90
    assert data.stored_bytes.min()>0
    assert (data["family"].str.contains('float16|float32')).sum()==0
    assert summary['n_paired'].eq(6).all()
    assert len(detail[['m','contrast','world','seed']].drop_duplicates())==90

def test_primary_fp64_contrasts():
    data=pd.read_csv(ROOT/'results/fp64_subset.csv')
    summary,_=audit(data)
    def diff(m,f):
        return summary.loc[(summary.m==m)&(summary.contrast=='mirror - '+f),'mean_difference_pp'].iloc[0]
    assert abs(diff(512,'plain')-(-1.1574074))<1e-5
    assert abs(diff(768,'plain')-(0.2777778))<1e-5
    assert abs(diff(768,'narrow_10')-(-0.5092593))<1e-5

def test_incomplete_data_refused():
    data=pd.read_csv(ROOT/'results/fp64_subset.csv')
    with pytest.raises(AssertionError):audit(data.iloc[1:])

def test_fp16_row_cannot_pollute_fp64_subset():
    data=pd.read_csv(ROOT/'results/fp64_subset.csv')
    fake=data.iloc[0].copy()
    fake['family']='teacher_float16'
    data=pd.concat([data,pd.DataFrame([fake])],ignore_index=True)
    with pytest.raises(AssertionError):audit(data)
