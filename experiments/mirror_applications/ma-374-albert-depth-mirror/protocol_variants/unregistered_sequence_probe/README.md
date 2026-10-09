# MA-374 — ALBERT shared Transformer block with depth Mirror coordinates

Status: PROMISING (development-only final-depth signal); multi-depth gate missed and fresh remains sealed.
Evidence lane: QUALITY / STORAGE / COMPUTE / DEPTH DIVERSITY
Base commit: `research/ma-368-factorized-width-depth-mirror-20261008`
Doctrine: `docs/phase2/MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md`
Integration map: `docs/phase2/MIRROR_PARAMETER_INTEGRATION_MATRIX.md`

## H — Hypothesis

One shared causal Transformer block plus one depth-specific Givens angle will restore per-depth next-token quality lost by ALBERT-style hard sharing, beyond four byte-matched scalar residual gates, while retaining much lower serialized bytes than untied layers.

## Mirror insertion

> **Mirror insertion:** this experiment adds one scalar angle after each repeated shared block application, rotating adjacent residual channels to express four depth roles without four separate Transformer blocks.

- Native method before adding `m`: ALBERT-style hard sharing of attention, FFN, and normalization across four applications.
- Exact insertion point for `m`: post-block residual stream at each logical depth.
- Persistent or dynamic `m`: four learned Givens angles.
- Mirror resources: fixed four-code coordinate; no sweep in this screen.
- Cheapest ordinary parameter: one scalar residual-branch gain per depth.

## Physical-to-logical claim

- Physical object being shared: one pre-norm causal Transformer block.
- Mirror/View coordinate: one Givens angle at each repeated depth application.
- Claimed logical multiplicity: four depth-conditioned functions using the same block weights.
- Why this could save storage: 4 angles replace the parameters in 3 additional blocks.
- Why it might fail: same-block recurrence may not require depth diversity; scalar gains may be sufficient; 4 rotations may not recover independently untied layer functions.

## Prior-art delta

Read the registry row and referenced PA items first.

- Closest prior art: PA61 ALBERT.
- What prior art already establishes: transformer attention/FFN parameters can be shared across layers.
- Exact Mirror-specific delta tested here: whether a low-description depth coordinate recovers measured per-prefix quality under hard sharing.
- Cheapest simpler control that could explain the result: four scalar residual gates.

## T — comparisons and protocol

Train untied, full ALBERT tying, attention-only sharing, FFN-only sharing, tied+scalar gate, and tied+Givens Mirror. All use the same recurrence-generated sequence world, 1,200 AdamW updates, and batch sequence for each development seed. Evaluate next-token NLL and accuracy at all four depth prefixes after loading paid serialized bytes.

## Gates

### PASS
Only if both development seeds pass the complete gate frozen in `PROTOCOL.json`; then fresh seeds 37411–37413 may open.

### FAIL
Mirror misses the registered final NLL, scalar-control, per-prefix, byte, or compute gate in both development seeds.

### NOT ESTABLISHED
Untied learnability gate or hard-tied gap does not establish a meaningful test; fresh stays sealed.

## Tuning boundary

Development: seeds 37401/37402; protocol and hyperparameters frozen.

Fresh: seeds 37411/37412/37413; never used for tuning.

## Storage contract

All embedding, attention, FFN, normalization, depth codes, readout, and metadata are paid. Deterministic ZIP/NPY FP16 archive size is authoritative.

## Compute contract

Record 1200 updates, tokens, training wall time, approximate active MACs per token, and evaluation throughput.

## Results

### D — decision

**Fact:** In both development worlds, untied validation accuracy exceeded 0.94 and hard ALBERT sharing had a final-NLL gap >0.02 nat. Mirror final validation NLL was 0.1906/0.1730 vs hard-tied 0.3163/0.2389 and same-byte scalar 0.2403/0.2052; test values were 0.1893/0.1742 vs 0.3107/0.2394 and 0.2352/0.2076. It was within 0.05 nat of untied in both worlds. Actual payload was 12,131B, equal to scalar and 70.3% below untied (40,821B). The depth-1..4 NLL improved over hard tying at only 1/4 prefixes in seed 37401 and 2/4 in seed 37402, so the preregistered development gate did not pass. All 12 payloads hash/size/reload/replay exactly; 4 tests pass. Fresh remains sealed.

**Interpretation:** This is PROMISING only for the final-depth function on this synthetic recurrence task: the four learned angles (0.014–0.082 radians) gave a repeatable final-NLL gain over both hard sharing and the equal-byte scalar control. The codes did not restore useful intermediate-depth behavior consistently. Attention-only sharing also matched untied quality in seed 37401 at 31,209B; FFN-only sharing was competitive in seed 37402 at 25,077B. Partial private parameters may be a better frontier for some worlds.

### C — strongest counter-hypothesis

The final-depth gain may be specific to this recurrence family or result from optimization/regularization effects rather than a generally useful Mirror depth role. The intermediate-prefix regression and lack of fresh confirmation leave this unresolved.

### U — unresolved

Fresh recurrence worlds, natural text, large Transformer behavior, near-convergence, and deployment kernels are untested. CPU throughput varied across seeds; no runtime gain is established. This is not an ALBERT or language-model claim and not a capacity result.

### Evidence labels

- **Fact:** the measurements above and per-prefix metrics in `results/development/`.
- **Interpretation:** final-depth quality improved under the tested training budget; depthwise quality recovery was inconsistent.
- **Hypothesis:** structured coordinates inside repeated attention/FFN could preserve intermediate depth functions better than a post-block rotation.
