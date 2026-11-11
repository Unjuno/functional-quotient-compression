# MA-800 — Vector-Network atoms with a persistent task coefficient code

Status: **FAIL for Mirror-specific value and registered storage gate**

## Prior-art boundary

PA213 describes reusable rank-1 weight atoms and per-input local inference. This synthetic screen includes a shared rank-1 atom bank and recomputes a top-2 atom selection for every input using `|v_kᵀx|`. It does not reproduce the published Vector Network training or inference algorithm.

## Fact

Seed 80001; 8 rank-1 atoms over 8D inputs/outputs; 9 tasks on a 3×3 factor grid; two diagonal tasks held out; 512 test examples/task; local inference selected exactly two atoms per input and visited 28 distinct atom subsets. Independent task coefficients use 1,851 B and have zero oracle error. Shared coefficients without task codes use 1,578 B but heldout nRMSE .7421. Factorized task coefficients use 2,076 B, heldout nRMSE 5.08e-8, and all-task nRMSE 3.44e-8. Mirror and native low-rank coefficient factorization have the same payload hash `9f367a7e593fdaadee31865feea24f2c124f9de8b7a489d56ab97124703ff8df` and identical metrics. The factorized bundle is 12.2% larger than independent despite using a low-rank teacher. Active MAC proxy is 80/example for all methods; decoding task coefficients adds 216 MAC to materialize all nine tasks. There are zero optimizer updates. Payload roundtrips are exact.

## Interpretation

The local atom selection is active and varies across inputs, while the task coefficient factorization recovers heldout compositions. Yet the factorized payload costs more actual bytes than independent coefficients because compressed NPZ exploits redundancy in the full coefficient table efficiently and the coordinate/basis metadata costs bytes. Even before storage, the Mirror representation exactly aliases ordinary native coefficient factorization. No compression or Mirror-specific frontier gain is established.

## H / T / D / C / U

- **H:** Factorized task coordinates over shared rank-1 atoms preserve top-2 local inference on heldout task compositions and reduce actual bytes by at least 25% versus independent task coefficient tables.
- **T:** Seed 80001; K=8, D=8; nine additive two-factor task compositions, two held out; 512 examples/task; input-local top-2 atom scoring; independent, shared-only, factorized View and native coefficient factorization. Measured actual NPZ bytes/hash, heldout/all-task output error, active atom sets, MAC proxies, local inference examples, wall time, zero updates.
- **D:** **FAIL.** Heldout quality and local atom selection pass the mechanism screen, but View storage is 12.2% larger than independent, misses the 25% savings gate, and exactly aliases the native low-rank factorization. Fresh stayed sealed.
- **C:** The task coefficient table is an additive low-rank function of two known task factors, so ordinary matrix factorization provides the same coordinates; the compressible full table also serializes efficiently.
- **U:** Trained Vector Networks, task-code inference, gradient learning, natural OOD compositions, larger banks/dimensions, quantized codebooks and GPU/fused inference.

This result is synthetic coefficient-factorization evidence only. It does not make a claim about trained Vector Networks.
