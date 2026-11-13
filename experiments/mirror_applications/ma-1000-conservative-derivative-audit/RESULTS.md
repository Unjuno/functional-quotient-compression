# MA-1000 — Conservative atomistic derivative audit

Status: **PROMISING for physical-constraint validity of this synthetic insertion only**. Compression/task value on a trained neural potential is **NOT ESTABLISHED**.
Branch: `research/ma-1000-conservative-derivative-audit-20261113`

## H / T / D / C / U

**H — Hypothesis:** placing a low-dimensional Mirror coordinate in a scalar invariant energy layer and deriving force after the View yields conservative, rotation-equivariant forces; a separately perturbed force head violates the constraint.

**T — Test:** CPU synthetic radial energy basis. Two different rank-2 codes per world were evaluated on seeds 100011/100013/100017. Compared gradient-derived Mirror energy, exact native rank-2 coefficient alias, independent full coefficients, and a separate-force-head negative control. No model fitting or optimizer updates; two functions per world. Actual serialized NPZ payloads include centers, width, bases, codes or full coefficient tables. Batch wall times were 0.085s, 0.034s, 0.023s; they are joint test times, not method-specific.

**D — Decision:** **PROMISING for conservative validity**: max finite-difference force-gradient residual 7.19e-11, rotation-equivariance residual 5e-16; deliberately separate force-head control residual 0.002. This does not establish an application or compression win. Mirror exactly aliases native rank-2 and costs 2,034 B versus 1,294 B independent (+57.2%); compute proxy is 110 vs 100 MAC/config.

**C — Strongest counter-hypothesis:** conservative forces follow from the definition `F=-∇E`; Mirror does not uniquely provide this property. The native low-rank coefficient control is functionally identical and smaller than the Mirror bundle in the selected serializer.

**U — Unconfirmed:** trained MACE/NequIP integrations, natural material energy/force quality, adaptation sample efficiency, MD stability, and any quality/byte Pareto gain.

## Facts

- All 6 Mirror code functions passed derivative and rotation checks in 3 fresh worlds.
- Maximum finite-difference residual: 7.19e-11; maximum equivariance residual: 5e-16; all negative-control residuals were .002.
- Mirror/native rank-2 payload: 2,034 B; independent coefficient payload: 1,294 B. Serialized arrays round-tripped exactly.

## Interpretation

The scalar-energy insertion is a valid way to preserve conservative forces in the tested toy basis. It is a structural validation result, not evidence that Mirror improves capacity or compression.

## Hypothesis for follow-up

MA-1001 should measure whether energy-derived Views remain stable over long dynamics, with the native shared potential and independently fitted potentials as controls. Any positive result must separate physical correctness from ordinary low-rank parameterization.
