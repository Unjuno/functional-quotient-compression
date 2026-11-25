# MA-1052 — TRACE selected LoRA modules with task-horizon Mirror coordinates

## H — Hypothesis

One MOMENT-1-base backbone with a TRACE-selected rank-2 LoRA basis and one shared ProjDown head can support four dataset×horizon forecasting roles. A two-scalar-per-selected-module task/horizon Mirror coordinate should improve audit macro MASE by at least 2% over both hard-shared TRACE and an equal-size diagonal rank-gain code, while staying within 5% of four independent TRACE specialists and using less adaptation storage.

## T — Experiment

**Fact.** Used pinned MOMENT-1-base (`5e44b0ea26376a176360f87831124e018f876d96`; 453,940,120-byte safetensors checkpoint) and official MOMENT implementation commit `38f7310ad594100747ca2a8357e9c7ca7d323e0e`. TRACE's Gated DSIC and ProjDown were implemented from arXiv:2503.16991v3 because an official TRACE code repository was not located. The checkpoint uses `d_ff=2048`, while TRACE Table 4 describes 3072.

**Fact.** Compared 4 roles: ETTh1/ETTh2 oil-temperature target (OT), each at H=192 and H=720, context 512. ETT chronological 12/4/4-month split and train-only normalization. Development used 8 H=192 and 7 overlapping H=720 validation origins per dataset. Audit used 15 disjoint H=192 and 4 disjoint H=720 origins per dataset. Seeds were 105201 (development) and 105202 (fresh audit). Each role received 30 optimizer updates, batch 4, and 120 training windows. Pooled arms had 30 joint updates over all four roles; four private specialists had 30 updates each. DSIC used 84 rank-2 gated LoRA locations, 8 masked validation-gradient samples per pruning round, 10 rounds pruning 8 gates each to retain 4. The shared DSIC selector used 640 validation examples; independent selectors used 160 per specialist. The shared selected basis was amortized across pooled conditions.

Controls were native MOMENT linear probing, native TRACE per-role specialists, pooled hard-shared TRACE, same-byte diagonal rank gains, and seasonal naive (period 24). The two non-Mirror pooled controls and Mirror branch started from the same selected TRACE basis/head state.

**Fact.** Environment: PyTorch 2.14.1+cpu, Transformers 4.54.1, four CPU threads, CUDA unavailable. The initial synthetic batch-1 profile measured forward 0.163s/backward 0.126s; the full dev and audit screens took 681s and 729s. All model training/evaluation ran in the container.

## D — Decision

**FAIL (scoped).** On canonical metrics replayed from serialized inference payloads, Mirror audit macro MASE was **1.063280**, compared with **1.063251** for hard-shared TRACE and **1.061715** for diagonal rank gains. Mirror was 0.003% worse than hard sharing and 0.147% worse than the simple diagonal control; it missed the preregistered 2% improvement gate. Development payload metrics were 1.532708 Mirror, 1.536298 hard share, and 1.535562 diagonal. The small dev gain did not transfer to audit.

The audit Mirror bundle was **471,985,837 bytes** including the common base checkpoint and config, versus **471,985,294 bytes** for hard share and **471,985,835 bytes** for diagonal gains. Mirror therefore added 543 bytes over hard sharing and 2 bytes over diagonal gains, without a quality advantage.

For context, the four independent TRACE adaptations used 46,230,674 adaptation bytes (500,171,743 bytes including one shared base) and had audit macro MASE 1.083612. The pooled Mirror adaptation state was smaller and scored better on the aggregate, but lost to an independent specialist by 5.1% on the ETTh2-H192 role. The native linear-probe reference had audit macro MASE 1.066250 and 358,621,758 adaptation bytes. These are secondary one-seed screening signals, not capacity claims.

## C — Strongest counter-hypothesis

**Interpretation.** At this rank and training budget, shared LoRA plus a shared forecast head already absorbs most cross-role regularity. A rank-plane rotation/gain contributes too little task-specific function to beat a two-scalar diagonal gate or hard sharing. The remaining ETTh2-H192 gap may need private adaptation, though this one-run signal is not sufficient to establish that boundary.

## U — Unconfirmed

- No near-convergence or fixed-byte frontier was measured; these fixed-update screens establish no capacity gain.
- Only one model seed, one target channel, two ETT datasets, and two horizons were used.
- The TRACE implementation is paper-derived rather than official source; update schedule/batch were scaled to fit the available CPU, and the checkpoint feed-forward width differs from the paper table.
- No GPU inference or runtime claim was established. The CPU p50/p95 measurements had high run-to-run spread and are descriptive only.
- **Serialization observation:** private TRACE and linear-probe metrics replayed directly from their saved files. For the three pooled arms, in-memory metrics and reloaded-payload metrics differed (maximum MASE difference 0.0847 development and 0.0447 audit). The serialized-payload metrics were therefore made canonical for byte/quality comparisons and replayed exactly a second time. The source of the pooled in-memory discrepancy was not isolated; both the discrepancy and canonical results are preserved in the run JSON and replay verification.

## Fact / interpretation / hypothesis

- **Fact:** Mirror failed both the hard-shared and diagonal-control quality gates on the fresh audit payload metrics and cost 543 bytes more than hard sharing.
- **Interpretation:** This insertion point did not produce a useful Mirror-specific Pareto gain in the tested screen.
- **Hypothesis:** Another task representation or a private residual may be needed for the ETTh2-H192 exception. This needs a separate candidate/protocol.

Detailed aggregates are in [RESULTS_CORE.csv](RESULTS_CORE.csv), per-role values in [RESULTS_PER_ROLE.csv](RESULTS_PER_ROLE.csv), payload hashes in `runs/payload_hashes.json`, and canonical serialized-payload metrics in `runs/{dev,audit}/payload_metrics.json`.
