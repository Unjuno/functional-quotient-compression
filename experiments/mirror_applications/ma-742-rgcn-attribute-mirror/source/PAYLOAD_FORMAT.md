# MA-742 inference payload format v1

The exact inference payload starts with eight charged bytes: `MA74`, format version, method code, coordinate rank and segment code. It is followed by the ordered little-endian FP32 tensor bytes. Tensor shapes/order and the fixed dimensions are defined by format version 1 in `run_experiment.py`; this runtime is common to every method. Segment 0 is the full inference model, segment 1 is the shared basis, and segment 2 is the relation-coordinate state. The segment payload headers are included in the reported byte totals.

`unpack_payload()` validates the header, reconstructs every named tensor, rejects truncation/trailing bytes, and must produce exact FP32 equality with the serialized model state. Hashes are SHA-256 over the complete segment-0 payload.
