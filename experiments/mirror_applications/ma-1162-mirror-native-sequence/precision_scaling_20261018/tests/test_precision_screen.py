import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import mpmath as mp
from precision_screen import (virtual_vector, virtual_one, quantize,
    phase_value_from_fixed_integer, sensitivity_rows, exact_teacher, target_offorbit, virtual_one_with_derivative)


def test_one_scalar_generates_distinct_views():
    with mp.workprec(256):
        vals=virtual_vector(mp.mpf(1)/7)
        assert len(vals)==8 and len({mp.nstr(v,30) for v in vals})==8


def test_mirror_high_precision_detection():
    with mp.workprec(1024):
        s=mp.mpf(1)/7
        eps=mp.power(2,-200)
        high=virtual_one(s+eps,0)-virtual_one(s,0)
        assert high!=0
        assert float(s+eps)==float(s)


def test_precision_scaling_remains_one_scalar():
    with mp.workprec(512):
        for p in (32,128,256):
            s,n=quantize(mp.mpf('0.365'),p)
            assert s==mp.mpf(n)/mp.power(2,p)
            assert 0<=n<(1<<p)


def test_fixed_phase_address_is_inactive_below_precision():
    with mp.workprec(400):
        a=phase_value_from_fixed_integer(123,32,96)
        b=phase_value_from_fixed_integer(124,32,96)
        assert a==b
        a=phase_value_from_fixed_integer(123,128,96)
        b=phase_value_from_fixed_integer(124,128,96)
        assert a!=b


def test_low_ulp_cannot_excite_low_freq_after_float64_cast():
    with mp.workprec(512):
        p=256
        s,n=quantize(mp.mpf('0.314159'),p)
        assert float(s)==float(s+mp.power(2,-p))
        a=phase_value_from_fixed_integer(n,p,p-2)
        b=phase_value_from_fixed_integer(n+1,p,p-2)
        assert mp.fabs(a-b)>mp.mpf('0.0001')


def test_offorbit_target_tangent_orthogonal():
    with mp.workprec(512):
        t=mp.mpf('0.3125')
        v=virtual_vector(t)
        off=target_offorbit(t,v)
        tangent=[virtual_one_with_derivative(t,j)[1] for j in range(8)]
        dot=sum((a-b)*d for a,b,d in zip(off,v,tangent))
        assert mp.fabs(dot)<mp.mpf('1e-90')


def test_hand_tangent_matches_high_precision_difference():
    with mp.workprec(256):
        s=mp.mpf('0.31875')
        h=mp.power(2,-70)
        for j in (0,3,7):
            _,analytic=virtual_one_with_derivative(s,j)
            numeric=(virtual_one(s+h,j)-virtual_one(s-h,j))/(2*h)
            assert mp.fabs(analytic-numeric)<mp.mpf('1e-35')
