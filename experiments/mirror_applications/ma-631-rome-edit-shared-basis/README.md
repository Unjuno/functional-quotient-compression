# MA-631 — ROME rank-one edit bank with shared view basis

Status: SCREENING; prior art PA135 ROME. Dedicated branch: `research/ma-631-rome-edit-shared-basis-20261103`.

## H

A family of rank-one ROME edits whose target deltas lie in a shared 2D basis and factor across two edit attributes can be stored as a shared basis plus small factor codes while retaining edit efficacy and locality. Mirror-specific value must beat ordinary low-rank and additive factor-table controls.

## T

Use a synthetic 32D MLP output layer with 16 fact keys arranged 4×4; diagonal pair edits are held out while all factor levels occur in the remaining 12. Target deltas are generated as `U[:,0]*a_i + U[:,1]*b_j`. Fit shared basis and factor tables only on the 12 visible edits. Compare independent rank-one updates, shared low-rank basis with per-edit coefficients, Mirror shared basis with two factor tables, and native additive factorized coefficients. Report edit efficacy, off-key locality leakage, complete serialized payload bytes including base layer and key IDs, and operation counts. Fresh seeds 63111–63113 remain sealed on native alias.

## D

Pending.

## C / U

Counter-hypothesis: this is ordinary low-rank ROME update tying plus additive factorization. Synthetic edit targets do not establish factual editing in a pretrained Transformer.
