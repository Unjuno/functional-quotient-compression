# MA-603 status

Status: PROMISING — Draw29, protocol and update selection were frozen before fresh audit query evaluation.

## H

A scalar input-conditioned Givens Mirror view on one shared matrix can recover a smooth dynamic linear-map family with a better payload or CPU latency point than four-basis CondConv.

## T

Frozen synthetic linear teacher; development worlds 3/5; audit worlds 11/17/23; model seeds 31/47/59; five model/control families. Settings in `PROTOCOL.json`.

## D

**PASS for the aligned synthetic mechanism gate.** Across three audit worlds, Mirror query MSE was 0.000009/0.000012/0.000013; CondConv4 was 0.000841/0.007475/0.001312; full dynamic was 0.002663/0.002321/0.003161. Actual payload bytes: Mirror 772, CondConv4 1,840, FiLM 1,784, full dynamic 5,600, static 448. Mirror latency was 0.111 ms vs CondConv4 0.041 ms; training wall 1.58 s vs 1.03 s. Analytic MACs/example: 144 vs 384. Teacher is intentionally Givens-aligned. See audit CSV and payload manifest.

## C

Mirror is given a teacher from its own transformation family; a more optimized native CondConv gate/basis or fused kernel may close the gap on natural dynamic functions.

## U

The teacher is deliberately Givens-aligned and synthetic; this does not establish general dynamic-weight capacity, natural convolution performance, or end-to-end optimized runtime.
