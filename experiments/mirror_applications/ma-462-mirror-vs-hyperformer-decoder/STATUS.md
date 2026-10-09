# MA-462 status

- Status: FAIL for Mirror-specific advantage; scoped aligned quality/compute result
- Branch: `research/ma-462-mirror-vs-hyperformer-decoder-20261009`
- Base commit: `10487ec4`
- Initial protocol freeze: `008712e4`
- A1 Fourier-control freeze: `7af3b858`
- Development selected 1,500 MLP updates
- Original fresh worlds: 46210–46212 × seeds 0–2
- A1 fresh worlds: 46220–46222 × seeds 0–2
- Results/registry/ledger: pending final commit

## H / T / D / C / U

- H: Fixed task embeddings decoded as Mirror Views yield lower payload and comparable quality to HyperFormer.
- T: Mixed adapter task family: 75% rotation-aligned and 25% off-orbit; identical fixed embeddings supplied to Mirror, MLP, affine, rank-two, Fourier, and private controls. Actual payloads measured.
- D (Fact): Original N32 Mirror: NRMSE 0.0691 / 73.16B per context; MLP: 0.1991 / 110.78B. A1 Mirror: 0.0663 / 73.16B; Fourier control: 2.55e-7 / 77.16B. Thus the Fourier payload is only 5.2% larger and is far more accurate.
- D (Interpretation): Mirror is a strong compute-efficient inductive bias on the aligned rotation portion, but fails the A1 Mirror-specific storage/quality gate against the generic Fourier control.
- C: A fixed trigonometric feature map matches the teacher geometry and explains the initial apparent Mirror advantage.
- U: Natural adapters and Transformer fine-tuning are untested.

## Next action

Commit both fresh result sets, update registry/ledger, push, then continue to MA-463.

## Blockers

None.
