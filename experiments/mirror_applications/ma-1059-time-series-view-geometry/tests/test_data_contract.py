import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from ma1059 import parse_tsf_line, windows


def test_tsf_parser_reads_single_selected_row():
    name,values=parse_tsf_line('T999:2000-01-01 00-00-00:1,2,3,4')
    assert name=='T999'
    assert np.array_equal(values,np.array([1,2,3,4],dtype=np.float32))


def test_monthly_windows_respect_registered_chronological_boundaries():
    x=np.arange(120,dtype=np.float32)
    w=windows(x)
    assert len(w['support'])==4 and len(w['query'])==1
    assert max(o+12 for _,_,o in w['support']) <= 72
    qctx,qy,qo=w['query'][0]
    assert qo==96 and len(qctx)==12 and len(qy)==12
