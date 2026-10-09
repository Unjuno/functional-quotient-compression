# MA-501 — Shared LoReFT subspace plus per-task coordinate

Status: **FAIL for Mirror-specific attribution and the rho=.25 quality gate**  
Branch: `research/ma-501-loreft-shared-mirror-code-20261008`  
Prior art: PA96 (ReFT / LoReFT), PA98 (activation addition)

## H — Hypothesis

In a frozen linear representation screen with sixteen task interventions, one rank-four shared subspace plus task-specific codes will preserve heldout intervention quality through private-variation ratio 0.25 at no more than half the actual serialized bytes of independent rank-four LoReFT-style interventions.

## Mirror insertion

**Mirror insertion:** this experiment adds a four-value task code `m_t` to a shared rank-four hidden-representation intervention so that sixteen task-specific linear functions can be expressed without storing separate rank-four factors for every task.

- Native method: task-specific low-rank representation intervention on frozen hidden state.
- Insertion point: after frozen hidden representation `h=x`, before fixed identity readout.
- `m_t`: persistent task coordinate stored with each function in the intervention bank.
- Cheapest ordinary control: native shared-subspace coefficients, algebraically identical to `m_t`; activation addition and per-task LoReFT-style factors are measured too.

## T — Executed protocol

Two development worlds (50101/50102); 16 tasks; hidden dimension 24; shared rank four plus private rank-four component at rho {0,.25,.5,1}. The shared basis is fit from task IDs 0–11 only. Task IDs 12–15 each receive 128 calibration examples for their code and are evaluated on 256 separate examples. We compare no intervention, activation addition, independent per-task rank-one and rank-four LoReFT-style factors, the shared code, exact native shared-code control, and full-matrix upper. Every intervention tensor, code, vector and metadata byte is included in the actual uncompressed NPZ payload. Fresh seeds 50111–50113 remain sealed.

## D — FAIL

At rho=0, shared-code relative RMSE was zero in both worlds and payload was 2,138 B versus 13,668 B for independent rank-four interventions (84.4% fewer bytes); per-task rank one had relative RMSE 0.402/0.478. Compute was equal to rank four at 196 operations/example and below the full-matrix upper at 576. At rho=.25, shared-code heldout task relative RMSE rose to 0.226/0.173, exceeding the frozen .10 limit; independent rank four was 0.178/0.161 and the full-matrix upper was exact. At rho=.5 and 1, shared-code error rose to 0.331–0.421 and 0.575–0.681 respectively. Activation addition and the no-intervention baseline were near relative RMSE 1 across conditions.

The native shared-subspace coefficient control has byte-identical payload and exact outputs for every seed and rho. The aligned compression is standard shared-basis low-rank intervention; there is no Mirror-specific gain. The private-variation gate fails and fresh seeds were not opened.

## C — Strongest counter-hypothesis

The rho=0 compression comes entirely from task functions sharing a rank-four LoReFT-style subspace. Once private directions appear, shared rank four is less accurate than independent task rank four; representing all rank-eight variation requires more private capacity. That residual crossover was not directly measured here.

## U — Still unconfirmed

This is a synthetic linear hidden-state mechanism screen, not a full ReFT/LoReFT training run or Transformer evaluation. No language quality, behavior steering, natural-task transfer, or production latency was measured. A rank-eight/private-residual frontier and alternative learned-subspace optimizer remain untested.

## Evidence classification

**Facts:** two frozen development worlds, actual serialized bytes, all rho curves, causal code perturbation, full-matrix upper and deterministic replay are recorded in `runs/`, `RESULTS_CORE.csv` and `VERIFICATION.json`.  
**Interpretation:** aligned sharing gives an 84.4% payload reduction but native shared coefficients exactly explain it; the rho=.25 quality gate fails.  
**Hypothesis:** natural task interventions may show another shared/private crossover, but MA-501 does not establish one.
