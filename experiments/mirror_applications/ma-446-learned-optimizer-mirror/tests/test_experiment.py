import csv
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))
from experiment import CoordinateLSTM, decode_m, generator, make_world, tensor_bytes


class MA446MechanismTests(unittest.TestCase):
    def test_basis_is_orthonormal_and_decodes_known_code(self):
        w0,basis=make_world(44601)
        flat=basis.reshape(3,-1)
        torch.testing.assert_close(flat@flat.T,torch.eye(3),atol=1e-6,rtol=1e-6)
        code=torch.tensor([[.2,-.5,1.1],[0.,0.,0.]])
        weights=decode_m(w0,basis,code)
        self.assertEqual(tuple(weights.shape),(2,3,4))
        torch.testing.assert_close(weights[1],w0)

    def test_coordinate_lstm_output_and_state_shapes_are_finite(self):
        model=CoordinateLSTM(hidden=8)
        grad=torch.tensor([[0.1,-2.0,4.0],[1.0,0.0,-0.3]])
        state=model.initial_state((2,3),grad)
        delta,next_state=model(grad,state)
        self.assertEqual(tuple(delta.shape),(2,3))
        self.assertEqual(tuple(next_state[0].shape),(2,3,8))
        self.assertTrue(torch.isfinite(delta).all())
        self.assertTrue(torch.isfinite(next_state[1]).all())

    def test_serialized_tensor_payload_round_trips_exactly(self):
        payload={'schema':'MA-446/test','params':torch.arange(12,dtype=torch.float32).reshape(3,4)}
        blob=tensor_bytes(payload)
        restored=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
        self.assertEqual(len(blob),len(tensor_bytes(payload)))
        torch.testing.assert_close(restored['params'],payload['params'],atol=0,rtol=0)

    def test_random_draw_pool_hash_and_index_replay(self):
        source=ROOT/'source'
        with (source/'selection_pool.csv').open(newline='') as handle:
            pool=list(csv.DictReader(handle))
        draw=json.loads((source/'random_draw.json').read_text())
        buf=io.StringIO(newline='')
        writer=csv.DictWriter(buf,fieldnames=list(pool[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(pool)
        self.assertEqual(hashlib.sha256(buf.getvalue().encode()).hexdigest(),draw['pool_csv_sha256'])
        seed=bytes.fromhex(draw['seed_hex']); limit=(1<<256)//len(pool)*len(pool)
        counter=0
        while True:
            value=int.from_bytes(hashlib.sha256(seed+counter.to_bytes(4,'big')).digest(),'big')
            if value<limit: break
            counter+=1
        self.assertEqual(counter,draw['rejection_counter'])
        index=value%len(pool)
        self.assertEqual(index,draw['index_zero_based'])
        self.assertEqual(pool[index]['id'],draw['selected_row']['id'])
        self.assertEqual(pool[index]['id'],'MA-446')


if __name__=='__main__': unittest.main()
