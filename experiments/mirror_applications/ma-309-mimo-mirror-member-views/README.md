# MA-309 — MIMO subnetworks with Mirror member views

Status: FAIL after corrected serialized-payload evaluation. Dedicated branch: `research/ma-309-mimo-mirror-member-views-20261008`.

## H — hypothesis

Four member-specific Givens views on a single shared MIMO trunk and classifier may preserve each synthetic task's accuracy and member prediction diversity while using fewer actual bytes than ordinary independent MIMO heads. A scalar output gate is the cheapest non-Mirror control.

## Mirror insertion

PA42's MIMO method obtains multiple functionally distinct subnetworks inside one physical network and evaluates their member outputs together. This experiment places one learned angle per member between the shared hidden trunk and shared output head. It compares standard MIMO member heads, hard tying, scalar output gates, Mirror views and independent MLPs.

## Frozen protocol

See `PROTOCOL.json`. Four binary classification rules on 2D Gaussian inputs; 32,000 train, 1,024 validation and 2,048 test examples/member; batch 64/member and 500 Adam updates. Development seeds 30901/30902; corrected fresh seeds 30921/30922/30923. Metrics reload the FP16 inference payload before evaluation.

## Development observations

After amendment A1 reloads, both development seeds remain: standard MIMO heads reached 0.976–0.984 macro accuracy, while Mirror reached 0.843. Mirror payload was 4,652B vs 6,256B (25.6% smaller); pairwise disagreement was 0.433–0.473 vs 0.500–0.506. The initial FP32-only fresh seeds 30911–30913 are quarantined and excluded.


## H/T/D/C/U report

**H — Hypothesis.** Four member Givens views on one shared MIMO trunk/classifier would preserve accuracy within 2 percentage points and disagreement within 0.05 of standard MIMO heads while reducing total payload by at least 15%; a scalar gate tests the cheapest simple control.

**T — Trial.** Four 2D binary rules, four members, shared 2->32->32 trunk and two-logit output. Standard MIMO heads, hard tying, scalar gate, shared-head Givens view and independent MLPs each trained for 500 Adam updates at batch 64/member. Dev seeds 30901/30902; corrected fresh seeds 30921/30922/30923. Metrics use 2,048 held-out examples/member and a common 4,096-point probe. The initial FP32-only fresh seeds were quarantined after implementation audit; A1 evaluates after reloading the actual FP16 ZIP/NPY payload.

**D — FAIL on accuracy and diversity.** Across fresh seeds, standard MIMO heads averaged 0.986 macro accuracy, 0.0356 NLL, 0.0068 ECE and 0.497 pairwise disagreement at 6,256B. Mirror averaged 0.809 accuracy, 0.3759 NLL, 0.1051 ECE and 0.189 disagreement at 4,652B (25.6% fewer bytes). Accuracy was lower by 15.0–19.7 percentage points and disagreement by 0.259–0.362 in every seed. Mirror CPU throughput was 0.62–0.85x standard MIMO. The scalar gate used the same 4,652B but averaged 0.625 accuracy and 0.333 disagreement; it was worse than Mirror, but Mirror still did not preserve the MIMO function quality/diversity point. Independent MLPs reached 0.991 accuracy and 0.499 disagreement at 16,264B.

**C — Strongest counter-hypothesis.** A Givens rotation on only two hidden coordinates may be too narrow to express the member-specific nonlinear rules. The byte saving is real but comes from deleting independent heads; the shared trunk does not make the view equally capable. The reduced disagreement indicates that explicit views collapsed some implicit MIMO diversity in this placement.

**U — Unverified.** This is a small synthetic task family, fixed-update learning-efficiency screen, and not a full reproduction of PA42. No natural dataset, trained language model, capacity frontier, calibration-under-shift or GPU kernel is tested.

### Evidence categories

- **Fact:** 15 corrected fresh payloads were reloaded and byte/hash checked; 15 metric rows replayed exactly (max difference 0); five tests pass. The initial FP32-only run is preserved under `artifacts/quarantined_pre_A1/` and excluded.
- **Interpretation:** In this insertion, a small member view compresses heads but loses too much member accuracy and prediction diversity. It does not improve the useful quality/diversity/bytes frontier over standard MIMO heads.
- **Hypothesis:** Views over a larger hidden subspace or learned member-specific low-rank adapters could preserve more MIMO diversity, but need a new experiment with the same measured controls.
