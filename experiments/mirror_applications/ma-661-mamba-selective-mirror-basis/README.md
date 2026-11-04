# MA-661 — Mamba selective-parameter Mirror basis

Screen a shared per-token coordinate for input-selective B, C and Delta generators in a diagonal SSM. A synthetic teacher uses input-dependent parameter projections. Compare the full native projections, separate native low-rank factors, a native joint input basis, and the same joint basis described as a Mirror view.

Measure sequence output NRMSE, actual inference payload bytes and a generator-MAC proxy across rank and off-basis residual. The native joint-basis control is required because `m(x)=Vx` may be ordinary shared low-rank factorization. This is not a Mamba reproduction; see `PROTOCOL.json`.
