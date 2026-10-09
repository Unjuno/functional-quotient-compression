# MA-545 — Few-shot-routed function-vector experts without weight experts

Status: FAIL
Evidence lane: LANGUAGE/REPRESENTATION_ROUTING/CAUSAL_QUALITY/ACTUAL_BYTES
Branch: `research/ma-545-fv-residual-moe-20261009`
Base commit: `a37cb993d0a9bef5c7a9bb39e86abca39ed30bf6`

## H — Hypothesis

A router reading few-shot demonstrations can select support-derived residual-stream FV experts on 16 Pythia-70M relation tasks, match oracle FV quality, and use <=0.75x the same-router rank-one MLP-output expert bytes, with useful held-out behavior.

## T — Execution

Pinned Pythia-70M-deduped (`e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`), CPU/five threads; development seeds 54501/54502, fresh seeds 54511–54513; 16 tasks, 8 support and 8 held-out queries per task. Each held-out prompt contains four support demonstrations. Per-task FV is the mean of eight layer-3 residual differences. One affine 512-to-16 router gets 1000 Adam updates and selects top-1. The rank-one weight control is a support-only ridge/SVD rank-one residual on the layer-3 MLP output. Fresh seeds were accessed after the development gate commit; no setting changed after access.

Protocol amendments: amendment 1 fixed NumPy/PyTorch RNG to the world seed; amendment 2 serialized the split manifest in each charged NPZ. Original unseeded development artifacts remain under `results/pre_amendment_1/`.

## D — FAIL for useful logical experts on the primary task metric

The preregistered router/oracle/byte/log-likelihood screening gates passed, but the simple shared-mean FV control beat the per-task routed experts on held-out candidate accuracy in all five seeds while using much less payload. Thus routing 16 logical function vectors did not improve the experiment's primary quality metric over one shared intervention.

## Facts

- Held-out route accuracy and macro-F1: **1.00** on both dev and all three fresh seeds; confusion matrices are diagonal with 8 examples per task.
- Routed FV candidate accuracy matches oracle FV exactly on every seed. Fresh accuracy is 0.3359, 0.3438, 0.3359.
- On fresh seeds, routed FV vs same-prompt ICL averages **+0.0469 accuracy** and **+0.2944 gold-logprob nats**.
- On fresh seeds, routed FV vs shared-mean FV averages **−0.0443 accuracy**, but **+0.2808 gold-logprob nats**. Shared mean uses **9,445 B** vs routed FV **78,117 B** (8.27x smaller); the routed FV lost candidate accuracy on all 3 seeds.
- Routed FV uses **78,117 B** vs same-router rank-one weight experts **111,109 B** (0.703x; 29.7% fewer bytes). Rank-one quality is nearly identical: fresh gold-logprob differs by at most 0.00075 nats and accuracy by at most 0.0078.
- Each fresh seed evaluates 1,024 candidate sequences and 43,152 tokens per method. Five-control evaluation takes 137.4–140.5 s; FV extraction takes 18.7–30.4 s; profiled router fit takes 0.60–0.65 s; rank-one fit takes 9.72–10.59 s. The final route audit adds 6.43–6.73 s.
- Full pinned model deployment is about 168.1 MB before these small task states. Incremental storage is the meaningful comparison here; no broad capacity claim follows from 16 route labels.

## C — Strongest counter-hypothesis

The four demonstrations make task identity obvious to the router and already supply the task mapping to ordinary ICL. The FV changes candidate likelihood, but one shared mean intervention gives higher answer accuracy with about one-eighth the routed-system payload. The near-identical rank-one control also explains the routed FV's likelihood behavior as ordinary low-rank conditional adaptation.

## U — Not established

Other model scales, unrelated task families, free-form generation, long-context routing, router robustness to noisy/missing demonstrations, and any useful increase in independent functional capacity remain untested. The result is a five-world, single-checkpoint screen.

## Storage / compute / quality frontier

Fact: FV routing beats the rank-one bank on incremental bytes at almost identical likelihood, but shared-mean wins the primary accuracy metric at a much smaller payload. Interpretation: there is a likelihood-versus-answer-accuracy tradeoff; this run does not show that per-task logical experts improve the accuracy/storage frontier. Hypothesis: a task-conditioned FV may be useful where calibrated gold likelihood matters more than top-1 candidate accuracy, but this requires a separate preregistered test.

See `PROTOCOL.json`, `RESULTS_CORE.csv`, `VERIFICATION.json`, per-seed `metrics.json`, and `route_audit.json`.
