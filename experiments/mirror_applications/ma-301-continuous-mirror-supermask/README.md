# MA-301 — Continuous Mirror supermask over a fixed backbone

Status: **FAIL** (amended development/fresh screen)
Prior art: PA31 SupSup and PA32 Piggyback/PackNet. Binary task masks are the mandatory native comparison.

## Hypothesis

A shared fixed random-feature backbone and readout plus one quantized Mirror coordinate per task can encode many logical task functions with fewer actual payload bytes than a per-task binary supermask while preserving near-independent quality.

> **Mirror insertion:** this experiment adds one task-specific angle `m_t` to a fixed pairwise Givens view over shared hidden features so that task readouts can differ without storing a full per-task mask or readout.

## Protocol and amendment

The first 4-task development screen serialized with `torch.save`; fixed ZIP overhead made Mirror and binary-mask payloads equal at 2,149 B, so its preregistered storage gate failed. No fresh data was opened. That variant is retained under `protocol_variants/`.

Before fresh access, the protocol was amended to 32 tasks and a deterministic compact binary payload. Mirror task angles are int8; binary task masks are bitpacked; all headers, fixed-encoder reconstruction metadata, shared weights, and private weights are charged. Development worlds 30110/30111 selected LR .003. Fresh seeds 30112–30114 were locked before access.

The five controls are hard tying, packed binary masks, fp16 continuous gates, Mirror angle coordinates, and independent readouts. Test functions are evaluated from deserialized payload state. Protocol: [PROTOCOL.json](PROTOCOL.json); source: [source/run.py](source/run.py); primary table: [RESULTS_CORE.csv](RESULTS_CORE.csv).

## Decision

The amended Mirror payload is 334 B, 0.788x the 424 B binary-mask payload. However, Mirror missed the independent quality threshold in two of three fresh worlds. The result is FAIL for the preregistered fixed-budget screen, not a capacity impossibility claim. The throughput column uses precomputed effective vectors and excludes coordinate construction, so it is not a direct-view runtime result.

See [STATUS.md](STATUS.md) for H/T/D/C/U, strongest counter-hypothesis, fact/interpretation/hypothesis separation, and limits.
