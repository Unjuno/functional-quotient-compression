# MA-501 — Shared LoReFT subspace plus per-task coordinate

Status: SCREENING; protocol pending freeze  
Branch: `research/ma-501-loreft-shared-mirror-code-20261008`  
Prior art: PA96 (ReFT / LoReFT), PA98 (activation addition)

## H — Hypothesis

In a frozen linear representation screen with sixteen task interventions, one rank-four shared subspace plus task-specific codes will preserve heldout intervention quality through private-variation ratio 0.25 at no more than half the actual serialized bytes of independent rank-four LoReFT-style interventions.

## Mirror insertion

**Mirror insertion:** this experiment adds a four-value task code `m_t` to a shared rank-four hidden-representation intervention so that sixteen task-specific linear functions can be expressed without storing a separate pair of rank-four factors for every task.

- Native method: a task-specific low-rank representation intervention on frozen hidden state.
- Insertion point: after frozen hidden representation `h=x`, before fixed identity readout.
- `m_t`: persistent task coordinate stored with the intervention bank.
- Cheapest direct control: ordinary shared-subspace coefficient intervention, algebraically matched to `m_t`; also compare activation addition and independent LoReFT-style factors.

## T — Frozen protocol summary

Sixteen seeded linear hidden-state functions in dimension 24; rank-four common intervention plus a controlled task-private rank-four component at rho 0, .25, .5 and 1. Basis fitting uses twelve task identities only. Four withheld task identities each receive 128 calibration examples for their own code, followed by 256 heldout inputs. Two development worlds (50101/50102); fresh 50111–50113 are sealed. Actual NPZ bytes charge every basis, code, factor, matrix, vector and schema.

This is a LoReFT-style mechanism screen, not a full Transformer or language-model evaluation. See `PROTOCOL.json` for exact definitions and frozen gates.
