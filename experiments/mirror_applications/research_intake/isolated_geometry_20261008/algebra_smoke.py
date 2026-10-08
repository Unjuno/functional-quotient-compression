#!/usr/bin/env python3
"""Pure-Python algebra invariants for experiments MA-1116/1120/1121 and MA-692.
These are small mathematical smoke tests only, not trained-model results.
"""
import math
import unittest


def mm(a, b):
    assert a and b and len(a[0]) == len(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def tr(a):
    return [list(row) for row in zip(*a)]


def inv2(a):
    (a0, a1), (a2, a3) = a
    det = a0 * a3 - a1 * a2
    if abs(det) < 1e-10:
        raise ValueError("singular")
    return [[a3 / det, -a1 / det], [-a2 / det, a0 / det]]


def err(a, b):
    return max(abs(x - y) for u, v in zip(a, b) for x, y in zip(u, v))


class TestAlgebra(unittest.TestCase):
    def test_attention_qk_gauge_is_function_preserving(self):
        q = [[1.2, -0.3], [-0.8, 0.5]]
        k = [[0.1, 0.7], [1.0, -0.5], [-0.3, 1.4]]
        g = [[1.2, -0.4], [0.3, 1.3]]
        expected = mm(q, tr(k))
        transformed = mm(mm(q, g), tr(mm(k, tr(inv2(g)))))
        self.assertLess(err(expected, transformed), 1e-12)

    def test_gva_content_keys_are_query_absorbable(self):
        q = [[0.4, 0.3], [-0.1, 0.8]]
        v = [[0.2, 0.5], [0.7, -0.4], [0.9, 0.2]]
        m = [[1.3, -0.3], [0.2, 0.9]]
        explicit = mm(q, tr(mm(v, m)))
        absorbed = mm(mm(q, tr(m)), tr(v))
        self.assertLess(err(explicit, absorbed), 1e-12)

    def test_rope_commutant_and_non_commutant(self):
        theta = 0.63
        rotation = [[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]]
        complex_scale = [[1.3, -0.2], [0.2, 1.3]]
        shear = [[1.0, 0.4], [0.0, 1.0]]
        self.assertLess(err(mm(rotation, complex_scale),
                            mm(complex_scale, rotation)), 1e-12)
        self.assertGreater(err(mm(rotation, shear), mm(shear, rotation)), 1e-3)

    def test_task_composition_can_be_noncommutative(self):
        rot90 = [[0.0, -1.0], [1.0, 0.0]]
        scale = [[2.0, 0.0], [0.0, 1.0]]
        self.assertGreater(err(mm(rot90, scale), mm(scale, rot90)), 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
