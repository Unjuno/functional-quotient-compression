import hashlib, json, pathlib, unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]

class ArtifactReplay(unittest.TestCase):
    def test_serialized_payload_bytes_and_hashes(self):
        art=ROOT/'artifacts'; rows=[json.loads(s) for s in (art/'runs.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),126)
        for r in rows:
            raw=(art/'payloads'/r['payload_file']).read_bytes()
            self.assertEqual(len(raw),r['serialized_bytes'])
            digest=hashlib.sha256(raw).hexdigest()
            self.assertEqual(digest,r['payload_sha256'])
            self.assertEqual(digest,r['roundtrip_sha256'])

    def test_locked_fresh_worlds_and_conditions(self):
        rows=[json.loads(s) for s in (ROOT/'artifacts/runs.jsonl').read_text().splitlines()]
        fresh=[r for r in rows if r['split']=='fresh']
        self.assertEqual({r['world'] for r in fresh},{40510,40511,40512})
        self.assertEqual({r['method'] for r in fresh},{'shared','style_mod_demod','mirror_givens','film','rank1','independent'})
        self.assertEqual({r['lr'] for r in fresh},{0.01})

if __name__=='__main__': unittest.main()
