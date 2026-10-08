from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import engine as e


class MA257Tests(unittest.TestCase):
    def test_support_split_covers_every_factor_level(self):
        for residues in (1, 2, 3):
            mask = e.support_mask(residues)
            self.assertEqual(int(mask.sum()), 64 * residues)
            for p in range(3):
                self.assertEqual(set(e.factors(int(i))[p] for i in np.flatnonzero(mask)), set(range(e.K)))

    def test_disjoint_contexts_compose_and_commute(self):
        z = np.arange(12, dtype=np.float64).reshape(2, 6)
        a = e.apply_plane(z, 0, np.cos(.3), np.sin(.3))
        ab = e.apply_plane(a, 1, np.cos(-.4), np.sin(-.4))
        ba = e.apply_plane(e.apply_plane(z, 1, np.cos(-.4), np.sin(-.4)), 0, np.cos(.3), np.sin(.3))
        np.testing.assert_allclose(ab, ba, atol=1e-14)

    def test_angle_and_free_coefficient_factorizations_are_equivalent(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(64, e.D)); base = rng.normal(size=(e.D, e.D))
        theta = np.array([[0., .2, -.3, .4, -.5, .6, -.7, .8]] * 3)
        a = b = c = 3
        z1 = x @ base
        z2 = x @ base
        for p, level in enumerate((a, b, c)):
            t = theta[p, level]
            z1 = e.apply_plane(z1, p, np.cos(t), np.sin(t))
            z2 = e.apply_plane(z2, p, np.cos(t), np.sin(t))
        np.testing.assert_array_equal(z1, z2)

    def test_serialized_payload_roundtrip_and_hash_replay(self):
        records = [("matrix", np.arange(12, dtype=np.float32).reshape(3, 4)),
                   ("half", np.array([.1, -.3], dtype=np.float16)),
                   ("byte", np.array([2, 9], dtype=np.uint8))]
        payload = e.pack_records(records)
        got = e.unpack_records(payload)
        self.assertEqual(payload, e.pack_records(records))
        self.assertEqual(e.hashlib.sha256(payload).hexdigest(), e.hashlib.sha256(e.pack_records(records)).hexdigest())
        np.testing.assert_array_equal(got["matrix"], records[0][1])
        np.testing.assert_array_equal(got["half"], records[1][1])

    def test_full_support_recovers_aligned_factor_codes_on_heldout(self):
        world = e.build_world(25711, "aligned")
        support = np.ones(e.K ** 3, dtype=bool)
        theta, _ = e.extract_factor_codes(world.weights[0], world.weights, support)
        for idx in range(e.K ** 3):
            a, b, c = e.factors(idx)
            pred = world.test_x[idx] @ world.weights[0]
            for p, level in enumerate((a, b, c)):
                t = theta[p, level]
                pred = e.apply_plane(pred, p, np.cos(t), np.sin(t))
            np.testing.assert_allclose(pred, world.test_y[idx], rtol=1e-10, atol=1e-10)

    def test_native_pa16_rotation_control_is_exact_mirror_alias(self):
        world = e.build_world(25701, "aligned")
        support = e.support_mask(1)
        estimates = np.zeros_like(world.weights)
        estimates[support] = world.weights[support]
        theta, coeff = e.extract_factor_codes(world.weights[0], estimates, support)
        zeros = np.zeros((e.K ** 3, 3), dtype=np.float64)
        p1 = e.build_payload("mirror_f16", world.weights[0], theta, coeff, zeros,
                             estimates, world.weights, support, world.seed)
        p2 = e.build_payload("native_rotation_f16", world.weights[0], theta, coeff, zeros,
                             estimates, world.weights, support, world.seed)
        self.assertEqual(len(p1), len(p2))
        s1, s2 = e.unpack_records(p1), e.unpack_records(p2)
        for idx in (0, 1, 63, 255, 511):
            np.testing.assert_array_equal(e.predict(s1, "mirror_f16", idx, world.test_x[idx]),
                                          e.predict(s2, "native_rotation_f16", idx, world.test_x[idx]))

    def test_psp_context_is_deterministically_reconstructed(self):
        np.testing.assert_array_equal(e.psp_context(19, 7), e.psp_context(19, 7))
        self.assertFalse(np.array_equal(e.psp_context(19, 7), e.psp_context(19, 8)))


if __name__ == "__main__":
    unittest.main()
