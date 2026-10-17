"""Deterministic numerical correctness checks, not new-model performance tests."""
import math
import random
import mpmath as mp
from precision_coordinate_probe import mirror_program, BITS, SEED


def test_exact_dyadic_grid_and_prefix():
    nstar=random.Random(SEED).getrandbits(BITS)
    for p in (16,32,53,64,128,256,800):
        n=nstar>>(BITS-p)
        child=nstar>>(BITS-p-1)
        assert child//2==n
        with mp.workprec(p+125):
            a=mp.mpf(n)/mp.power(2,p)
            b=mp.mpf(n+1)/mp.power(2,p)
            c=mp.mpf(child)/mp.power(2,p+1)
            assert b-a==mp.power(2,-p)
            assert c-a in (0,mp.power(2,-p-1))


def test_float64_collapse_of_fine_grid():
    nstar=random.Random(SEED).getrandbits(BITS)
    for p in (64,128,256,800):
        n=nstar>>(BITS-p)
        with mp.workprec(p+125):
            a=mp.mpf(n)/mp.power(2,p)
            b=mp.mpf(n+1)/mp.power(2,p)
            assert a!=b and float(a)==float(b)


def test_four_step_mirror_responds_to_exact_fine_changes():
    nstar=random.Random(SEED).getrandbits(BITS)
    for p in (32,64,128,256,800):
        n=nstar>>(BITS-p)
        with mp.workprec(p+125):
            a=mp.mpf(n)/mp.power(2,p)
            b=mp.mpf(n+1)/mp.power(2,p)
            ya,yb=mirror_program(a),mirror_program(b)
            assert ya != yb
            sensitivity=abs((yb-ya)/(b-a))
            assert 4 < sensitivity < 5


def test_integer_only_training_variable():
    nstar=random.Random(SEED).getrandbits(BITS)
    p=128
    n=nstar>>(BITS-p)
    assert isinstance(n,int)
    assert isinstance(p,int)
