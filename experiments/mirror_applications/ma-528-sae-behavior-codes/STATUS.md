# MA-528 status

- Status: SCREENING
- Branch: `research/ma-528-sae-behavior-codes-20261009`
- PA102 reviewed; protocol and controls frozen before implementation and development.
- Frozen protocol hash is in `freeze.json`.
- Development seeds 52801/52802 not yet run; fresh seeds 52811–52813 locked.
- MA-526 prior failure and MA-527 Givens failure are controls/context, not substituted outcomes for this distinct residual-selected feature-basis experiment.

## H / T / D / C / U

- **H:** residual-selected shared 64-atom SAE pool and 16-term behavior codes can preserve explicit-FV quality, modestly compress global SAE OMP16 lists, and beat activation-frequency pool selection.
- **T:** pinned Pythia-70m and layer-3 SAE; feature pool selected on task vectors 0–11 only; tasks 12–15 held out; two dev seeds; explicit FV, global OMP16 and matched aggregate-pool OMP16 controls.
- **D:** screening; no development metrics accessed.
- **C:** a shared basis may not contain unrelated function-vector directions; amortized pool/index storage may erase savings, while global OMP may dominate quality.
- **U:** quality, actual bytes, compute, deterministic replay and heldout attribution remain unknown.
