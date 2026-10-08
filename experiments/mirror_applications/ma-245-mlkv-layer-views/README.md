# MA-245 — MLKV shared cache + per-layer Mirror KV views

Status: PROMISING (quality/cache frontier; model-payload gate missed)
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `1c313c1ac6aba2b5c4cd4bcc3933c08b1f98ce19`

## H — hypothesis

If layer-specific K/V functions are related by low-description role/layer coordinates, one physical cached K/V pair plus layer-specific Mirror addresses can restore distinct attention outputs while using fewer cache bytes than multi-group MLKV and outperforming a byte-near scalar gate.

## T — what ran

PA08 (*MLKV: Multi-Layer Key-Value Heads for Memory Efficient Transformer Decoding*) shares KV heads across layer groups and reports KV cache size proportional to layer groups and head groups. This mechanism screen used four methods plus controls: four independent per-layer K/V projections (MHA), two-group MLKV, one-group hard-tied MLKV, one-group Mirror MLKV, and same-size scalar Gate MLKV.

The teacher used one shared K projection and one shared V projection, with a separately sampled Givens angle for each layer and role. Four layer queries attended to the same eight-token memory. Students predicted the four teacher attention outputs. Common LR `0.003` was selected on development world 24500 only; three fresh worlds (24501–24503) independently sampled projections, role/layer angles, and examples. Every method trained 900 updates with the same data budget.

Actual serialized state-dict plus config bytes were measured. K/V cache payload bytes were measured from materialized tensors (`numel × element_size`). Active MAC proxies, wall times, and one-thread CPU throughput are in `RESULTS_CORE.csv` and `VERIFICATION_REPLAY.json`.

## D — decision

**PROMISING for the aligned output/cache mechanism; strict model-payload gate missed.** Mirror MSE was `6.7e-12`–`6.8e-11` in all three fresh worlds, at or below the independent-layer MHA result and about nine orders below one-group hard MLKV and the byte-near scalar gate. The gate's payload was 2,539 bytes versus Mirror's 2,541 bytes, but its MSE was `0.015`–`0.026`.

Mirror used 2,541 serialized bytes versus 2,864 for two-group MLKV (11.3% fewer), missing the preregistered 25% payload reduction. It used 256 actual cache bytes versus 512 for two-group MLKV (50% fewer) and 1,024 for MHA (75% fewer). One-group hard MLKV used 2,352 model bytes and the same 256 cache bytes, but its MSE was `0.016`–`0.028`; the Mirror's eight paid angle codes restore layer-specific views over that shared pair.

The eager CPU implementation was slower: Mirror training took about 2.1–2.2x two-group MLKV, and median measured inference throughput was 0.85x. All 15 fresh rows replayed; maximum MSE difference was `3.5e-12`, and payload/cache bytes matched. Tests: 4 passed.

## C — strongest counter-hypothesis

The teacher is generated exactly from a shared base plus the same Givens views being tested. All four layers also receive the same memory hidden state. This deliberately aligned setup isolates view reconstruction and cache accounting, but omits layer-to-layer hidden-state changes and does not establish that pretrained Transformer layers share this geometry.

## U — what remains unconfirmed

- Whether natural-language transformer layer K/V maps have compact shared coordinates.
- Whether a cache-view implementation can reuse state across layers when their memory hidden states differ.
- Whether a fused view kernel recovers MLKV throughput.
- Perplexity, autoregressive generation, bandwidth, GPU runtime, and near-convergence fixed-byte frontier.

## Fact / interpretation / hypothesis

- **Fact:** the MSE, serialized payload bytes, measured cache tensors, updates, MAC proxy, wall time, and replay differences follow the frozen protocol.
- **Interpretation:** layer-specific views recover a deliberately aligned teacher from one shared cached K/V pair and halve cache bytes relative to two-group MLKV, while eager compute regresses and actual model-payload reduction misses its threshold.
- **Hypothesis:** naturally trained layers may contain low-description KV relationships that allow a similar cache/quality frontier; this remains untested.
