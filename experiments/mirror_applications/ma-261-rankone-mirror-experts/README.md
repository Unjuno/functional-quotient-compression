# MA-261 — BatchEnsemble-style rank-one Mirror experts

Status: SCREENING (protocol frozen)
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `f7f76de193063950d28b7834f337840b1e86f0ce`
Prior art: PA17 BatchEnsemble.

## H — falsifiable hypothesis

A shared physical expert matrix plus expert-specific Mirror coordinates can recover two logical routed functions from one matrix with lower serialized bytes than independent experts, and outperform the mandatory rank-one BatchEnsemble expert modulation on held-out examples at comparable bytes and active compute.

## Mirror insertion

> **Mirror insertion:** this experiment adds one low-description Givens coordinate `m_e` to the shared expert's hidden representation so two logical routed expert functions can be expressed without duplicating the expert matrix.

- Physical object: one 16x16 expert matrix.
- Coordinate: expert-ID keyed hidden Givens rotation.
- Logical objects: two top-1 routed experts, with an oracle router shared across methods.
- Native control: shared matrix plus rank-one input/output fast factors (BatchEnsemble).
- Controls: hard-tied single expert, independent full experts, BatchEnsemble rank-one experts.

## Fixed task and gates

The router selects expert 0 for the first eight input coordinates and expert 1 for the last eight. Teachers are 16D-to-8D linear maps restricted to the matching half. Development worlds 26100/26101; LR {0.003, 0.01}, 1,200 AdamW updates, batch 128. Fresh 26102–26104 remain sealed unless dev passes.

**PASS (screen only):** Mirror MSE <=1.10x independent, actual payload <=65% independent, and beats BatchEnsemble by >=10% MSE at byte-near cost (within 15%) or strictly lower bytes. **FAIL:** misses quality/storage gate or BatchEnsemble matches at equal/lower bytes.

Report oracle routing explicitly, task coverage, expertwise MSE, serialized bytes, MAC proxy, wall time and throughput.

## Boundaries

Synthetic oracle-routed regression only; not a learned-router MoE result, no natural language, no capacity claim from fixed updates.

## D — FAIL (development screen)

Development selected LR 0.003. In both fixed development worlds, BatchEnsemble rank-one factors fitted the teacher nearly exactly (MSE about 1e-10), and independent experts reached <=1e-8. Mirror Givens stayed near hard tying at MSE about 1.8 despite using 2898 bytes versus independent 3227 bytes. It therefore failed the preregistered quality gate; fresh worlds 26102–26104 remain unopened.

### C — strongest counter-hypothesis

The teacher's two experts are separated by input-half structure, which rank-one input/output factors can directly encode. The registered Mirror transform only rotates within the halves and may not express the needed routing-conditioned mask; this is a geometry mismatch, not evidence that every Mirror expert view fails. Oracle routing was supplied to all methods and remains an external aid.

### U — not established

Fresh-world replication, learned router behavior/load balance, natural-language quality, and alternative Mirror coordinate families remain untested.

### Fact / interpretation / hypothesis

- **Fact:** Two development worlds x two learning rates x four methods were run for 1,200 updates; actual serialized payloads were measured. Selected LR is 0.003; BatchEnsemble and independent controls fit near machine precision, Mirror did not.
- **Interpretation:** This Givens insertion failed to replace the registered rank-one expert modulation on the fixed oracle-routed task.
- **Hypothesis:** Cross-half transforms or input-conditioned views might be required to express this task's expert masks; a new MA/amendment would be needed to test them.
