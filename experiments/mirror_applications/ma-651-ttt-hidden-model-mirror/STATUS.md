# MA-651 status

- Status: FAIL
- Branch: `research/ma-651-ttt-hidden-model-mirror-20261104`
- Base commit: `7f086dae`
- Protocol frozen before development: yes; one development-only LR amendment recorded in protocol/freeze
- Development complete: yes (65101, 65102; LR=0.3)
- Fresh/audit opened: no; native-ID alias gate reached

## H / T / D / C / U

**H:** A shared bank of low-rank writable output subspaces plus one code per stream preserves long-context online adaptation while reducing recurrent hidden-state bytes versus full TTT-Linear; private/mismatched modes expose the state-capacity boundary.

**T:** TTT-like supervised linear regression: 16 streams, 4 rank-4 output modes, 128 tokens/stream. Two development seeds. Compared full TTT-Linear, one fixed rank-4 basis, per-stream private rank-4 basis, shared basis dictionary selected from an 8-token bootstrap gradient, and native dictionary IDs. All recurrent state and shared bases were serialized. A shared LR grid selected 0.3 on development final-window MSE; no fresh data accessed.

**D:** FAIL for Mirror-specific attribution. Shared dictionary state is 7,115–7,179B (41.5–41.7% of full TTT-Linear state at 17,117–17,238B), with correct 100% mode routing and identical prequential output to private rank-4 updates. Final-32-token MSE is .00108–.00110 for dictionary/private versus .00139–.00140 full. However, the private rank-4 state is smaller at 6,944–6,990B, and Mirror exactly equals native dictionary IDs. A single fixed basis is smaller (6,211–6,232B) but has final-window MSE .0129–.0134, about 9–10x higher. Fresh seeds 65111–65113 remained sealed.

**C:** The improvement over full TTT is low-rank inductive bias aligned with the task outputs; private bases compress better under actual NPZ serialization than a shared bank plus code. Mirror is just ordinary dictionary routing. The synthetic task may also favor known rank-4 structure.

**U:** This is a supervised linear TTT-like task, not the paper's self-supervised TTT-Linear/MLP. No language-model NLL, realistic token dynamics, or learned dictionary construction was tested. Initial target orientation and LR=1.0 attempts were implementation/development screens preserved separately; final results use corrected output orientation and frozen LR=0.3.

## Fact / Interpretation / Hypothesis

- **Fact:** 10 final development rows; all metrics and actual state bytes replay; three tests pass; protocol hash matches the amended freeze manifest; fresh untouched.
- **Interpretation:** Online low-rank hidden states can sharply reduce state bytes for this aligned task, but sharing the dictionary is not better than private rank-4 state and offers no Mirror-specific benefit over native IDs.
- **Hypothesis:** More streams or larger hidden dimensions may amortize shared dictionary overhead; a natural TTT task would need to show this over a strong private low-rank control.
