# MA-510 — Conditional activation Mirror steering

Status: FROZEN_SCREENING  
Evidence lane: MECHANISM / STORAGE / FALSE-TRIGGER  
Base commit: `48990a5a` (integrated through MA-508)  
Prior art: PA101 (Conditional Activation Steering)

## Hypothesis

H: Replacing explicit CAST behavior vectors with a shared rank-4 basis and compact Givens behavior codes reduces activation-steering bytes while preserving conditional efficacy and the false-trigger rate on held-out condition-behavior pairs.

## Mirror insertion

> **Mirror insertion:** this experiment adds a behavior-specific Givens View `m_j` to one shared activation-steering seed, so CAST's condition gate can select among logical behavior directions without storing a full activation vector per behavior.

- Native CAST: continuous context vector matched to stored condition keys; on match, add the selected behavior steering vector.
- Shared object: one 64×4 representation basis plus one rank-4 seed vector.
- View: two Givens angles per behavior; FP16 state is charged.
- Conditions and behavior keys remain identical across controls, isolating representation compression from routing.
- Generic nearest control: shared basis plus per-behavior FP16 coefficient table.

## Frozen protocol

- D=64, rank=4, 16 behaviors, 32 condition prototypes, 8D context codes.
- Target behavior vectors lie on a fixed-norm two-plane Givens orbit; `rho=.1` adds coefficient-space private residual.
- Each condition prototype has continuous Gaussian query noise (sigma=.2). CAST activates when cosine(query, behavior-key) > .35. This threshold and the rule labels are fixed by the protocol.
- 20% of condition-behavior pairs are held out for evaluation. False-trigger rate is primary; also report missed-trigger rate, intervention efficacy, vector NRMSE and off-target drift.
- Development: 50800/50801 × seeds 0/1/2; validates deterministic encoder only, no tuning. Fresh: 50820/50821/50822 × seeds 0/1/2.

## Controls

1. CAST explicit behavior vectors with shared condition prototypes/keys and cosine gate.
2. Generic shared basis + FP16 behavior coefficients, same condition state and gate.
3. Mirror shared basis + seed + FP16 Givens angles, same condition state and gate.
4. Mirror plus FP16 per-behavior residual, measuring private fallback cost.

## Gates

- PASS if every aligned fresh world has false-trigger rate ≤5%, miss rate ≤5%, efficacy ≥0.99, off-target drift ≤0.01, and Mirror bytes ≤80% of the cheapest generic basis-coefficient control.
- PROMISING if it compresses CAST explicit vectors and preserves gate quality but generic codes match it or compute is unfavorable.
- FAIL if Mirror violates conditional quality or does not beat generic coefficients by the 20% byte margin.

All condition prototypes, behavior keys, vectors/codes, bases, seeds, residuals and metadata are serialized and charged. The synthetic behavior geometry is intentionally aligned to the Mirror family; no pretrained-LM claim is made.
