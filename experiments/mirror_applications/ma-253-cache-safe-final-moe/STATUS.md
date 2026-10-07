# MA-253 status

- Status: SCREENING
- Branch: `research/ma-253-cache-safe-final-moe-20261007`
- Base commit: `01b515cf8d0603481cb00ff1d5be45541411eebb`
- Last verified commit: preregistration pending commit
- Development complete: yes; selected common LR `0.003`
- Fresh/audit opened: no
- Results committed: no
- Verification committed: no
- Registry row updated: no

## Next action

Freeze the protocol/source and run fresh worlds 25301–25303 at 1,500 updates.

## Blockers

None. CPU PyTorch is available; GPU is not required for this synthetic screen.

## Decisions / rulings

- PA14's final-layer placement is modeled as a feed-forward update after that layer's K/V projections.
- Development selected a common learning rate using dev MSE only. The independent rank-2 DMoE control fits nearly exactly; the Mirror-only coordinate is visibly underfit in development.
- A deterministic two-layer cache probe passed the preregistered final-only invariance and early-placement invalidation checks before fresh evaluation.
