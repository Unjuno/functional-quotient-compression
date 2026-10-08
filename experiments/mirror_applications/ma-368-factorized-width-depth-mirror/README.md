# MA-368 — Factorized width × depth Mirror code

Status: FAIL; frozen development gate did not pass.
Evidence lane: QUALITY / STORAGE / COMPUTE / UNSEEN CONFIG GENERALIZATION
Base commit: `research/ma-367-universally-slimmable-width-mirror-20261008`
Doctrine: `docs/phase2/MIRROR_PARAMETER_INTEGRATION_DOCTRINE.md`
Integration map: `docs/phase2/MIRROR_PARAMETER_INTEGRATION_MATRIX.md`

## H — Hypothesis

An additive width/depth Givens coordinate (four width values + four depth values) will improve four architecture combinations withheld from code training over ordinary US-Net/OFA-style shared weights and a byte-matched factorized scalar gain.

## Mirror insertion

> **Mirror insertion:** this experiment adds factorized coordinates `m_width + m_depth` to the final hidden activation of a shared max-width, max-depth MLP, so that configuration-specific function changes can be expressed without per-configuration weights.

- Native method before adding `m`: nested shared elastic supernet with US-Net sandwich sampling.
- Exact insertion point for `m`: adjacent-channel Givens rotation after the last active hidden block and before shared readout.
- Persistent or dynamic `m`: persistent, eight scalars total (4 width + 4 depth); angle is their sum.
- Mirror resources: fixed 8-scalar factorization, selected before development; no sweep or private residual in this screen.
- Cheapest ordinary parameter: same 8-scalar factorized multiplicative hidden gain.

## Physical-to-logical claim

- Physical object being shared: one 32-wide, 4-layer MLP and readout.
- Mirror/View coordinate: additive width/depth angles.
- Claimed logical multiplicity: 16 width-depth subnetworks from one parameter set.
- Why this could save storage: only 8 codes beyond the shared supernet instead of 16 independent subnetworks.
- Why it might fail: width/depth truncation errors may be unrelated to a single final rotation; an ordinary gain may do as well; withheld combinations may not generalize compositionally.

## Prior-art delta

Read the registry row and referenced PA items first.

- Closest prior art: PA51 US-Net and PA52 Once-for-All.
- What prior art already establishes: one shared supernet can serve multiple widths/depths through sandwich sampling/progressive shrinking.
- Exact Mirror-specific delta tested here: a small factorized functional code for unseen width-depth combinations.
- Cheapest simpler control that could explain the result: factorized scalar gain with the same eight learned values.

## T — comparisons and protocol

Compare independent per-configuration MLPs, ordinary shared elastic supernet, factorized scalar gain, and factorized Mirror. Train all shared variants with identical sandwich configs/minibatches. Hold out (8,3), (16,4), (24,1), (32,2) from shared-code updates and evaluate all 16 configurations.

## Gates

### PASS
Both dev worlds must pass the locked gate in PROTOCOL.json before fresh seeds open. Fresh success requires held-out NLL advantage, measured bytes and MAC within the frozen budgets, and exact serialized replay.

### FAIL
Mirror fails to improve held-out NLL over both ordinary supernet and factorized scalar control in both development seeds, or misses registered quality/storage/compute limits.

### NOT ESTABLISHED
Independent learnability control fails its gate or run/replay integrity is incomplete. Fresh stays sealed.

## Tuning boundary

Development: seeds 36801/36802; protocol is frozen. No hyperparameter changes after starting development.

Fresh: seeds 36811/36812/36813; never used for tuning.

## Storage contract

All shared tensors, width/depth codes, all independent model weights, metadata, and archive headers are charged. Deterministic ZIP/NPY FP16 payload size is authoritative.

## Compute contract

Record examples, 1,200 updates, per-configuration MACs, training wall clock, and inference examples/second.

## Results

### D — decision and result

**Fact:** Both independent references passed learnability (validation macro accuracy 0.852/0.829). On held-out configuration validation NLL, Mirror scored 0.5515/0.5805 versus plain supernet 0.5054/0.5210 and factorized scalar 0.5496/0.5588. Mirror was worse than both controls in both seeds. Test held-out NLL was 0.5621/0.5791 versus supernet 0.5148/0.5059 and scalar 0.5554/0.5488. The Mirror payload (12,645 bytes) equaled scalar, was 4.0% larger than plain supernet (12,161 bytes), and 83.9% smaller than the independent bank (78,685 bytes). Fit time was 2.37/2.54 s for Mirror versus 1.47/1.86 s supernet and 1.84/1.81 s scalar. All 8 payloads passed hash/size/reload and per-configuration metric replay; 3 unit tests passed. Fresh remains sealed.

**Interpretation:** On this synthetic shared-teacher task, factorized width/depth Givens angles did not generalize useful corrections to held-out architecture pairs. The native supernet already did better, and a scalar control was also better despite the same payload size. Mirror's storage reduction versus independent specializations comes from weight sharing already present in the supernet; it is not attributable to the added code.

### C — strongest counter-hypothesis

Held-out combinations may need configuration-specific calibration that an additive final-layer angle cannot express. A richer width/depth interaction code or correction inserted at each block could perform differently, but would need to beat the corresponding ordinary factorized control and pay all bytes/compute.

### U — unresolved

No natural workload, unseen continuous widths/depths, fused-kernel runtime, long convergence, or progressive-shrinking OFA implementation was tested. The result is a fixed-budget synthetic screen, not a capacity claim. MA-369 tests subnetwork-specific OFA correction as a separate candidate.

### Evidence labels

- **Fact:** measurements and execution details above and in `results/development/`.
- **Interpretation:** the tested additive Givens parameterization failed its held-out quality gate and showed no Mirror-specific frontier gain.
- **Hypothesis:** block-local or explicit interaction coordinates may be more useful; they remain untested.

## Decision

FACT:
INTERPRETATION:
HYPOTHESIS:
BOUNDARY:
