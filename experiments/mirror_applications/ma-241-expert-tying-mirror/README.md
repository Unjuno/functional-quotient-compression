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

Fact: see `RESULTS_CORE.csv` and `VERIFICATION.json`.  
Interpretation: this mechanism screen measures whether the specified coordinate can recover this deliberately aligned synthetic teacher.  
Hypothesis: broad generalization to naturally trained expert banks remains untested.
