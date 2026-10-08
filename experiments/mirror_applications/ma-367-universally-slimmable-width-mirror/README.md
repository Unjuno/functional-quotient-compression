# MA-367 — Universally slimmable network with width Mirror corrections

Status: FAIL; frozen development gate did not pass. Dedicated branch: `research/ma-367-universally-slimmable-width-mirror-20261008`.

## H — Hypothesis

A shared nested-width MLP with one learned Givens angle per width should recover some narrow-width quality while preserving full-width behavior. It must beat a byte-near direct scalar gate and stay within an explicit compute budget.

## Mirror insertion

PA51 US-Net already shares weights across widths. Here the physical object is one width-32 MLP with nested channel slices at widths 8, 16, 24, and 32. Mirror adds one width-specific Givens coordinate at the hidden activation/readout interface. A direct scalar activation gain uses the same four-code budget; IA3 per-channel gains are a stronger, larger ordinary control.

## T — Frozen protocol

A deterministic synthetic 32-input, four-class teacher MLP supplies 8,192 train, 2,048 validation, and 4,096 test examples per world. Shared models use sandwich-style width sampling: always train widths 8 and 32, plus one uniformly sampled intermediate width, with identical minibatches and sampled widths across variants. Compare independent models per width, plain shared US-Net, direct scalar-gain US-Net, Mirror-angle US-Net, and width-specific IA3 gains. Train for 1,000 Adam updates at 1e-3. Development seeds are 36701/36702; fresh seeds 36711–36713 remain sealed unless all development gates pass.

All inference states are deterministic ZIP/NPY archives with FP16 weights. The archive includes all shared weights, codes, width IDs, and metadata. Evaluate each width independently after reloading its actual serialized payload. Record per-width NLL/accuracy, bytes, active MACs, training wall time, and inference throughput.

## Controls and interpretation

- Independent width models provide a reference quality point and pay for every width's parameters.
- Plain US-Net measures ordinary nested weight sharing.
- Scalar gain is the matched low-description non-Mirror control.
- IA3 width gains show whether diagonal conditioning closes the gap with a larger ordinary code.

Logical width count is never treated as capacity. A positive result requires actual per-width quality and a Mirror-specific margin against simple controls.

## Gates

The exact development and fresh gates, storage contract, and fresh IDs are frozen in `PROTOCOL.json`. If the development gate fails, fresh remains unopened. Any configuration change after fresh access requires a separate amendment/ID.

## Decision record

### H — hypothesis and verdict

**Hypothesis:** One learned width-specific Givens angle would improve narrow-width quality beyond US-Net and a direct scalar gate while retaining full-width quality at comparable payload/compute. **Verdict: FAIL at the registered development gate.** No fresh data were opened.

### T — training and controls

**Fact:** Two development worlds (seeds 36701, 36702), 1,000 Adam updates per method, 8,192 training examples, sandwich width sampling, and five methods were run. Every actual FP16 ZIP/NPY payload was reloaded and replayed. Both seeds' hashes and validation/test metrics reproduced exactly. Three unit tests passed. See `results/development/` for recorded metric summaries.

### D — decision

**Fact:** Independent reference validation macro accuracy exceeded 0.70 on both seeds (0.739/0.806), satisfying the learnability gate. Mirror narrow-width NLL did not beat plain US-Net by 5%: seed 36701 width-8/16 validation NLL was 0.7258/0.6378 vs 0.6712/0.6024; seed 36702 was 0.5492/0.4820 vs 0.4978/0.4560. It also did not stay within 0.03 nat of independent at every width. Mirror was exactly payload-size tied with scalar gate at 4,179 bytes; mean narrow validation NLL was better than scalar on seed 36701 (0.6818 vs 0.7006) but worse on seed 36702 (0.5156 vs 0.4960; lower is better), far short of the registered 10% margin. The comparison against US-Net fails on both seeds. Mirror uses 4,179 vs 3,949 bytes (+5.8%), nominal active MACs are 5.6% higher, and measured training time was about 1.4x US-Net.

**Interpretation:** In this small nested-width task, Givens rotation after ReLU does not repair width truncation/interference. The activation rotation adds compute and no measured useful width-specific behavior beyond ordinary weight sharing. This is a synthetic fixed-budget negative result, not a near-converged capacity statement.

### C — strongest counter-hypothesis

The issue may be insertion-specific: a post-ReLU paired rotation cannot recover features omitted by nested channel truncation, and a broader width/depth coordinate or learned width-conditioned normalization may behave differently. The native US-Net already performs well, so the remaining gap may be too small for any four-scalar correction to improve.

### U — unresolved

Natural image or language workloads, unseen widths, alternate Givens placement, convergence, and hardware-fused rotation latency are untested. MA-368 separately tests width×depth factorization. Fresh MA-367 seeds remain unopened by design.

### Evidence labels

- **Fact:** reported measurements and protocol execution above; Givens MAC-equivalent count is 2 per adjacent channel pair (4 multiplies plus 2 additions).
- **Interpretation:** the selected placement did not recover truncation loss under this task/budget.
- **Hypothesis:** moving the view before truncation or introducing width×depth factorization may change the result; this requires another preregistered experiment.
