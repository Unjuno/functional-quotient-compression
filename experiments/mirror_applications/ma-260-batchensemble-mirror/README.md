# MA-260 — BatchEnsemble rank-one Mirror ensemble

Status: SCREENING (protocol frozen)
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `f7f76de193063950d28b7834f337840b1e86f0ce`
Prior art: PA17, BatchEnsemble.

## H — falsifiable hypothesis

On a synthetic multi-class task family, a shared physical feature map with learned Givens Mirror member views will improve ensemble diversity and predictive quality per serialized byte over native BatchEnsemble rank-one member factors, while retaining independent ensemble quality with fewer bytes. If BatchEnsemble matches the Mirror frontier, Mirror-specific value is falsified.

## Mirror insertion

> **Mirror insertion:** this experiment adds a compact member coordinate `m_k` to the shared feature map output, expressing diverse ensemble member predictions without storing one full feature matrix per member.

- Physical object: shared 16x16 feature projection and shared 16x4 classifier.
- Mirror coordinate: member-specific 8-plane Givens rotations in hidden space.
- Logical multiplicity: four ensemble members.
- Native method: BatchEnsemble shared matrix with member-specific rank-one input/output factors.
- Simple controls: one shared predictor (replicated as members), independent member MLPs, BatchEnsemble.

## Protocol and gates

Synthetic 4-class classification from 16D inputs. Four members, hidden width 16, 1,000 AdamW updates, batch 128. Development worlds 26000/26001; learning rates 0.003/0.01 selected by ensemble NLL. Fresh worlds 26002–26004 are sealed pending dev pass.

**PASS (screen only):** Mirror ensemble NLL within 10% of independent, ECE no worse by >0.03 absolute, payload <=60% independent, and beats BatchEnsemble by >=10% NLL or yields >=20% higher member disagreement at byte-near cost. **FAIL:** quality/storage gate missed or BatchEnsemble matches at equal/lower bytes.

Report ensemble and member accuracy, NLL, ECE, pairwise disagreement, actual serialized bytes, training examples/updates, MAC proxy and wall/throughput.

## Boundaries

Synthetic classification only; no natural-data ensemble or OOD calibration claim. Fixed-update screen is not a capacity proof.

## Development result — FAIL / harness invalidated

Two development attempts were run. The first used a complex random teacher and method-dependent input seeds; code audit found that train/test data depended on each method's initialization seed, so those rows are retained only as invalidated diagnostics. Before any fresh access, the task was amended to balanced separated Gaussian prototypes and all methods were rerun with common per-world data.

On that corrected data-seed run, dev NLLs remained highly variable across the two worlds (world 26000: independent 2.40, BatchEnsemble 3.20, Mirror 2.70 at LR .003; world 26001: independent 1.25, BatchEnsemble 2.21, Mirror 2.02). Member disagreement was exactly zero for BatchEnsemble and Mirror in all corrected dev rows, indicating member collapse under this initialization/optimization setup. The shared baseline also failed badly in one world. Common-LR selector chose 0.003, but the control regime was not stable enough for a meaningful ensemble frontier. The screen is therefore recorded as FAIL / harness did not establish an interpretable Mirror-versus-BatchEnsemble comparison. Fresh worlds 26002–26004 were never opened.

### C — strongest counter-hypothesis

The zero disagreement likely reflects symmetric initialization/optimization of member factors and inadequate diversity pressure, rather than a general limit of BatchEnsemble or Mirror. The teacher prototype scale and world-specific sampling also produced large inter-world difficulty differences.

### U — not established

No conclusion about BatchEnsemble versus Mirror diversity-per-byte, calibration, or independent ensemble quality. A valid follow-up needs a fixed teacher across worlds, verified learnability by every control, member-specific initialization that breaks symmetry, and explicit independent teacher/member functions.

### Fact / interpretation / hypothesis

- **Fact:** Development used two worlds, two learning rates, 1,000 updates and actual serialized payload measurements. Fixed fresh worlds remained sealed. The finalized run had exact zero member disagreement for Mirror and BatchEnsemble.
- **Interpretation:** The harness failed to produce a usable ensemble comparison, so this is not evidence of Mirror failure or success. It is a candidate-level FAIL to establish the registered claim under the screen.
- **Hypothesis:** Shared optimization symmetry suppressed member diversity; distinct member init/teacher variation may restore it in a future amended experiment.
