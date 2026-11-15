#!/usr/bin/env python3
"""Independent audit-payload and exactness-record checks for MA-1042."""
import json
import pathlib
import struct
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDIT = ROOT / "runs" / "audit"
MAGIC = 0x4D41313034325047
DOMAINS = 4
N = 16384
NOPS = 256


def read_header(path, expected_count):
    raw = path.read_bytes()
    magic, version, count = struct.unpack_from("<QQQ", raw)
    assert magic == MAGIC and version == 1 and count == expected_count
    return raw


def read_keys(path):
    raw = read_header(path, N)
    assert len(raw) == 24 + 8 * N
    return list(struct.unpack_from(f"<{N}Q", raw, 24))


def read_ops(path):
    raw = read_header(path, NOPS)
    assert len(raw) == 24 + 9 * NOPS
    return [(raw[24 + 9*i], struct.unpack_from("<Q", raw, 25 + 9*i)[0]) for i in range(NOPS)]


def payload_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


class MA1042PayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(x) for x in (AUDIT / "results.jsonl").read_text().splitlines()]

    def test_three_fresh_worlds_are_exact(self):
        self.assertEqual([r["seed"] for r in self.rows], [1050, 1051, 1052])
        for r in self.rows:
            self.assertEqual(r["epsilon"], 8)
            self.assertEqual(r["queries_total"], 40000)
            for method in ("mirror_exact", "linear_exact", "independent_exact"):
                self.assertEqual(r[method], r["queries_total"])
            self.assertEqual(r["no_view_exact"], 0)

    def test_reported_payload_matches_actual_files(self):
        for r in self.rows:
            world = AUDIT / f"seed_{r['seed']}"
            self.assertEqual(payload_bytes(world / "shared"), r["shared_payload_bytes"])
            self.assertEqual(payload_bytes(world / "independent"), r["independent_payload_bytes"])
            self.assertEqual(r["shared_payload_bytes"], 133720)
            self.assertEqual(r["independent_payload_bytes"], 534260)

    def test_shared_state_reconstructs_each_domain_and_update_stream(self):
        for r in self.rows:
            world = AUDIT / f"seed_{r['seed']}"
            base = read_keys(world / "shared" / "initial_keys.bin")
            ops = read_ops(world / "shared" / "updates.bin")
            view_raw = (world / "shared" / "views.bin").read_bytes()
            self.assertEqual(struct.unpack_from("<QQ", view_raw), (MAGIC, DOMAINS))
            views = [struct.unpack_from("<QQ", view_raw, 16 + 16*d) for d in range(DOMAINS)]
            self.assertEqual(views, [(3, 1_000_000_000), (5, 4_000_000_000), (7, 9_000_000_000), (11, 16_000_000_000)])
            for d, (scale, offset) in enumerate(views):
                root = world / "independent" / f"domain_{d}"
                keys_d = read_keys(root / "initial_keys.bin")
                ops_d = read_ops(root / "updates.bin")
                self.assertEqual(keys_d, [scale*k + offset for k in base])
                self.assertEqual(ops_d, [(kind, scale*k + offset) for kind, k in ops])

    def test_payload_reduces_three_quarters_and_latency_gate_holds(self):
        for r in self.rows:
            self.assertLessEqual(r["shared_payload_bytes"] / r["independent_payload_bytes"], 0.5)
            self.assertLessEqual(r["mirror_p95_ms"] / r["independent_p95_ms"], 1.10)
            self.assertLessEqual(r["mirror_p95_ms"] / r["linear_p95_ms"], 1.10)


if __name__ == "__main__":
    unittest.main()
