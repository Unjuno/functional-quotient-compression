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

## Development screen and early decision

### H / T / D / C / U

**H:** Shared condition/behavior factor codes plus Mirror behavior Views reduce CAST activation-state bytes while preserving held-out condition efficacy and false-trigger rate.

**T:** Fixed 64D rank-4 behavior-vector bank, 16 behaviors, 32 continuous condition prototypes, 8D behavior keys, cosine threshold .35, query noise sigma .2, 20% pair holdout; two development worlds × three seeds × two residual regimes; explicit CAST vectors, generic FP16/FP32 coefficients, Mirror FP16 angles, and Mirror plus private residual.

**D: FAIL at the development gate; fresh was not opened.** On `rho=0` held-out pairs, all representations had false-trigger 4.68% (within 5%) but miss rate 28.88% (gate ≤5%), giving on-target efficacy 71.12%. Mirror used 5,145B vs generic FP16 coefficients 4,957B and had slightly higher NRMSE (.000357 vs .000214); the generic control is both smaller and more accurate. Explicit CAST vectors used 7,713B. At `rho=.1`, Mirror-only NRMSE was .1458; private residuals reduced it to .000405 at 5,525B. Development failure is sufficient under the frozen stop rule; no fresh results are claimed.

**C:** The native cosine gate is brittle to context noise for near-threshold positives; because every representation shares that gate, the 28.9% miss rate is not caused by Mirror. Generic FP16 coefficients also dominate Mirror on the aligned behavior bank.

**U:** Fresh replication, a calibrated CAST threshold/gate, natural conditions, pretrained-LM behavior scores, and learned condition/behavior code factorization remain untested.

### Facts

- 60 development rows: 2 worlds × 3 seeds × 2 residual regimes × 5 methods.
- `rho=0`: FPR .0468, miss .2888, efficacy .7112 for all methods because the gate is shared.
- At `rho=0`, generic FP16 coefficient payload 4,957B / NRMSE .000214; Mirror 5,145B / .000357; explicit CAST 7,713B.
- `rho=.1`: Mirror-only NRMSE .1458; private residual variant .000405 at 5,525B.
- All development payload hashes, lengths and replay outputs verified exactly.

### Interpretation

A development-only gate failure means this protocol cannot support a passing conditional-steering claim. The representation comparison separately shows no Mirror-specific byte advantage over generic FP16 coefficients. The failure is preserved and the queue advances.
