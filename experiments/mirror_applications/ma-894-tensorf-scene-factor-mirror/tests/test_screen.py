import io
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
import run_screen as screen


class ScreenTests(unittest.TestCase):
    def test_generation_is_deterministic(self):
        a = screen.make_truth(89401)
        b = screen.make_truth(89401)
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_mirror_scene_code_equals_native_four_way_cp(self):
        train, fresh, x, y, z, codes = screen.make_truth(89401)
        mirror = screen.cp_eval(x, y, z, codes[screen.N_DEV:])
        native = torch.einsum("xr,yr,zr,sr->sxyz", x, y, z, codes[screen.N_DEV:]) + 0.5
        torch.testing.assert_close(mirror, native, rtol=0, atol=0)
        self.assertEqual(mirror.shape, fresh.shape)

    def test_coordinate_split_is_exact_partition(self):
        obs, held = screen.coordinate_split(89501)
        self.assertEqual(obs.numel() + held.numel(), screen.N ** 3)
        self.assertEqual(torch.unique(torch.cat([obs, held])).numel(), screen.N ** 3)
        self.assertEqual(torch.isin(obs, held).any().item(), False)

    def test_serialized_payload_roundtrip(self):
        arrays = {"x": np.arange(8, dtype=np.float32), "code": np.ones((2, 4), dtype=np.float32)}
        payload = screen.bytes_npz(arrays, {"rank": 4})
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["x"], arrays["x"])
            np.testing.assert_array_equal(archive["code"], arrays["code"])
            self.assertTrue(len(payload) > 0)


if __name__ == "__main__":
    unittest.main()
