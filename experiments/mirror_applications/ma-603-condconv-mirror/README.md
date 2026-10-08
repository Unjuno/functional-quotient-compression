# MA-603 — Input-conditioned Givens views versus CondConv

Status: PROMISING
Base commit: `c935a903daca5c7d1d48aa50d05b5bd50f239cba`  
Prior art: PA121 (CondConv), PA122 (Dynamic Filter Networks)

## H — falsifiable hypothesis

For an input-conditioned teacher whose linear map moves along a known Givens orbit, one shared matrix plus a learned scalar Mirror angle can recover the continuous family with useful quality and a better payload or CPU latency point than finite four-basis CondConv. Full dynamic matrix generation is an upper reference; FiLM and a static map test simpler controls.

## T — frozen protocol

Eight-dimensional standard-normal inputs contain a two-dimensional condition slice. The teacher applies a smooth scalar angle derived from that slice to four paired output Givens planes around a fixed world-specific matrix. Two world seeds are development-only; three fresh world seeds and three model seeds form the audit. Five methods are compared: Givens Mirror, CondConv4, FiLM8, a full dynamic matrix hypernetwork, and a static linear map. See `PROTOCOL.json` for the frozen architecture, optimizer, seed split, gates and metrics. Draw29 replay is in `source/draw29_exclusions.json`.

This is a controlled mechanism test, not a convolution/Transformer benchmark. The teacher is intentionally aligned to the Mirror transform.

## D — decision

**PASS for the preregistered aligned synthetic mechanism gate; operational status PROMISING.** On 3 fresh worlds × 3 model seeds, Mirror mean query MSE was `1.2e-5`, versus `0.00321` for CondConv4 and `0.00272` for the full dynamic hypernetwork. It beat both controls in all three worlds and used 772 serialized bytes, versus 1,840 B for CondConv4 (−58.0%) and 5,600 B for full dynamic. The static model was smallest (448 B) but had MSE 0.318; FiLM used 1,784 B and had MSE 0.142. Mirror had 137 parameters and 144 analytic active MACs/example; CondConv4 had 404 parameters and 384 MACs/example. Its CPU batch-1 latency was slower, 0.111 ms versus 0.041 ms for CondConv4, and mean training wall was 1.58 s versus 1.03 s. The result improves the storage/quality/analytic-MAC point on this intentionally Givens-aligned screen, while runtime overhead remains.

All 45 seed-level measurements and actual safetensors payload hashes are retained in `artifacts/metrics_audit.csv` and `source/payload_manifest.json`. The repeated audit run preserved the query MSE and payload values; wall/latency measurements varied as expected. The teacher explicitly applies the same family of Givens views used by Mirror, so this is a feasibility result for aligned dynamic functions, not evidence of general CondConv replacement or capacity increase.

## C — strongest counter-hypothesis

The teacher is deliberately constructed from the same Givens transformation used by Mirror. A native CondConv implementation with a better gate, more expert bases, or a fused kernel could close the measured quality/runtime gap; this screen does not establish a Mirror-specific gain on naturally occurring dynamic kernels.

## U — boundaries

No claim about natural dynamic kernels, language modeling, large-scale quality, GPU runtime, or Mirror-specific novelty follows from a synthetic aligned teacher.
