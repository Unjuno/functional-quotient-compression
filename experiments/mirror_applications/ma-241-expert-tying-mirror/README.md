# MA-241 — expert tying across depth + layer-specific Mirror views

Status: SCREENING  
Evidence lane: MECHANISM / STORAGE / RUNTIME  
Base commit: `ccf4d5c4e83992d70ccdc5db6032e428f6532380`

## Hypothesis

**H:** When two layers use related versions of the same expert pool under different low-description feature coordinates, a shared expert pool with learned layer-specific orthogonal views recovers most of the held-out function quality of untied experts with fewer serialized inference bytes; it must also beat a byte-near layer-specific low-rank residual to support a Mirror-specific claim.

## Physical-to-logical claim

- Physical object: a four-expert, two-layer MLP pool.
- View coordinate: four learned Givens rotation angles per layer; each expert is evaluated as `R_l^-1 E(R_l x)`.
- Logical multiplicity: two layer-specific expert behaviors from one shared physical expert pool.
- Storage opportunity: share expert matrices and pay only the per-layer angles and separate routers.
- Main risk: realistic layer differences may not be conjugate rotations; a small low-rank residual or ordinary tying may work as well.

## Prior-art delta

- Closest prior art: PA01, *Tying the Loop — Tied Expert Layers in Mixture-of-Experts Language Models*.
- Prior art establishes that expert weights can be tied across consecutive Transformer layers while routing remains layer-specific.
- Exact delta: test whether a small learned orthogonal coordinate on the expert function recovers layer-specific behavior lost by hard tying.
- Strong simple control: the same tied expert pool plus per-layer rank-2 expert-output residuals. A gate-only tied model is also measured.
- Scope: a synthetic regression mechanism screen, not a nanoGPT/language or capacity result. The synthetic teacher is intentionally generated from layer-coordinate transforms, so this is an optimistic feasibility case, not a neutral estimate of real-world prevalence.

## Comparisons

1. Untied MoE: independent four-expert pools and routers for both layers.
2. Ordinary expert tying: one pool, independent layer routers.
3. Tied + layer gate: one scalar per layer and expert on the expert output.
4. Tied + low-rank control: one rank-1 output residual per layer and expert.
5. Tied + Mirror: one pool, independent routers and four Givens angles per layer.

All methods train from scratch on the same teacher-generated split and update count. The target generation is deterministic from the world seed and is not serialized in any inference payload.

## Gates

### PASS for useful sharing

Across all three fresh worlds, Mirror held-out MSE is at most 1.10x untied MSE, at most 0.80x ordinary tied MSE, and actual inference payload is at least 20% smaller than untied.

### Mirror-specific signal

At comparable serialized bytes (within 10%) and active MAC proxy (within 10%), Mirror held-out MSE is at least 10% lower than the stronger of the gate and low-rank controls in every fresh world.

### FAIL

If Mirror misses the useful-sharing gate, or a simple tied control is within 10% MSE of Mirror at comparable or lower bytes, record a FAIL for the Mirror-specific claim. A failed mechanism screen does not establish failure on language data.

### NOT ESTABLISHED

Missing fresh worlds, non-reproducible serialization, training divergence, or a control that cannot be reproduced.

## Tuning boundary

- Development world: `24100`, split seed `241000`, initialization seed `2410000`.
- Fresh worlds: `24101`, `24102`, `24103`; their teacher functions and samples are generated only after the development run and configuration are frozen.
- Allowed development adjustment: choose one learning rate from `{1e-3, 3e-3}` using only development validation MSE, common to all methods.
- Fixed after development: dimensions, architecture, batch size, update count, optimizer, coordinate parameterization, rank, and fresh seeds.

## Storage and compute contract

Measure actual `torch.save` serialized state-dict bytes, including every learned tensor and layer-view code. Report config metadata bytes separately and include them in the total inference payload. MAC proxy counts router linear layers, expert MLPs, coordinate transforms and residuals; soft routing evaluates all four experts in every method. Report examples, updates, wall time, and CPU thread count.

## Results and decision

### H — hypothesis

For layer variants produced by low-description orthogonal coordinate changes, tied expert weights plus layer-specific Givens views recover most untied function quality with lower serialized storage, and beat byte-near gate/low-rank controls.

### T — execution

Five methods (untied, tied, gate, rank-1 residual, Mirror) trained 1,800 AdamW updates on the same generated examples per world. Common LR `0.003` was selected from two values on development world 24100 only. Fresh worlds 24101–24103 used independent teacher parameters, samples, and fixed initializations. CPU PyTorch 2.10.0, one thread, Xeon Platinum 8573C; all four experts were evaluated using soft routing. Actual `torch.save` state-dict bytes plus config JSON were counted. Each fresh result was independently replayed.

### D — decision

**PASS for the preregistered synthetic quality/storage mechanism gates; PROMISING at MA status level.** In all 3 fresh worlds, Mirror MSE was at most 1.10x untied and below 0.80x hard tying. Mirror payload was 24,286 bytes versus 45,681 untied (46.8% smaller) and 23,967 hard tied. At effectively matched payloads, Mirror MSE beat the gate by 26.6–66.1% and the rank-1 residual by 25.0–65.0% in the fresh worlds. All 15 quality rows replayed within `4.6e-13` absolute MSE difference.

**Compute/runtime did not improve in this implementation.** The analytical MAC proxy is within 2% of simple tied controls, but measured training was 2.1–2.3x the tied condition in the replay run. Inference examples/sec were 0.43x, 0.77x, and 0.61x tied across the three worlds (median 0.61x). This likely reflects unfused coordinate operations and Python/PyTorch overhead; no optimized kernel was tested, so MACs do not describe the observed latency.

### C — strongest counter-hypothesis

The synthetic teacher was intentionally constructed as a shared expert pool conjugated by the same type of learned orthogonal views. This is an aligned feasibility case. The apparent Mirror advantage may disappear for independently learned Transformer experts or for less structured layer changes. The simpler controls were not granted this exact teacher transform, while the Mirror was; this asymmetry makes the quality result optimistic for the Mirror family.

### U — unconfirmed

- Whether naturally trained layer-specific experts have recoverable shared coordinate structure.
- Whether a fused/compiled Givens implementation can keep the measured quality while avoiding current runtime overhead.
- Sparse top-k routing, Transformer/nanoGPT integration, natural-language quality, near-convergence capacity, and fixed-byte frontier.
- Whether private residual parameters are necessary when the layer differences include genuinely unrelated functions.

### Evidence classification

- **Fact:** the listed MSE, byte, update, MAC-proxy, throughput, and replay measurements were observed under this protocol.
- **Interpretation:** the View recovers a deliberately aligned layer variation in this small regression task and reduces weight storage, but the current implementation loses compute/runtime Pareto position.
- **Hypothesis:** pretrained MoE expert banks may contain similarly compact layer-coordinate structure; this experiment does not establish that.
