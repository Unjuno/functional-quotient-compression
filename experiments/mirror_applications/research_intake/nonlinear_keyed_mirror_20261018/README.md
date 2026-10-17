# NKM-INTAKE-001 — repeated nonlinear Mirror / conditional keys / functional gates

**Status: IDEA_REGISTERED / NO NEW TRAINED EXPERIMENT. Date: 2026-10-18 (JST).**
Research branch: `research/nonlinear-keyed-mirror-20261018`. Formal backlog candidates: **MA-1156..MA-1161** (all UNTESTED); these IDs are different hypotheses, not six completed experiment results.

> **Mirror insertion:** add small `m` to the activation-coordinate and optional condition interfaces of one shared FFN `theta`. Apply `Q(psi,m)`, nonlinear `phi_eta`, optional inverse/next `Q` repeatedly to obtain distinct functional rules without storing a full FFN for every logical role. Separately test a condition code `c` for routing and standard encrypted storage of `m` under secret `k`.

## Why this is different from ordinary weight sharing

The target functions may be far apart under the original input/output coordinates but close under a low-description group action. The proposed utility is **cheap coordinate-orbit factorization**, not ordinary weight/activation similarity.

Core operator on a hidden preactivation `z`:

```text
N_m(z) = Q(psi,m)^(-1) phi_eta(Q(psi,m) z)
N_m^(t+1)(z) = N_{m,t+1}(N_m^t(z))
```

With `phi=identity`, adjacent invertible Q factors often cancel or fold into one map. With nonlinear `phi`, the operations generally do not commute. However some transforms still commute with particular activations. These are explicit **null controls**.

The low-description code `m` is not a secret key. The condition `c` selecting a View is not cryptographic authorization. Encryption of `m` protects at-rest bytes only under a standard cryptographic and runtime threat model; it does not hide a plaintext shared `theta`.

## What is already known

- **DMFFN-001:** factorized gate/up Mirror helps on Givens-aligned teachers, while independent tasks and eager CPU runtime fail.
- **AMA-001:** learned activation and partial-private development experiment was negative; fresh was not opened.
- **MA-333/546:** non-trivial parameter differences may be symmetry-only.
- **MA-701/773:** reflection families and compositions exist in registry, but are not evidence for this nonlinear repeated operator.
- **MSR001:** fully tying expert View and router codes loses to a separate linear router; new gating is not implicitly vindicated.
- **MA-792/793:** routing-free activation is already a related candidate, not a new claim of novelty.
- **MLP-ROLE/MLP-USE:** strong plain task-readout controls have dominated some natural multi-role tasks.

## Artifacts and work discipline

- [program design and six hypothesis cards](../../../../docs/phase2/NONLINEAR_KEYED_MIRROR_RESEARCH_2026-10-18.md)
- [experiment blueprint](EXPERIMENT_BLUEPRINT.md)
- [security/condition separation](SECURITY_BOUNDARY.md)
- [pre-experimental draft constraints](PROTOCOL_DRAFT.json)
- [current status](STATUS.md)

The official registry records ideas as UNTESTED. Before any independent formal MA experiment, create a scoped `ma-1156-...` etc. experiment directory, freeze its precise development/fresh protocol, run tests, and **never** infer an MA PASS from this design-only commit.

No changes to `main`, historical FAIL/PROMISING results, Claim Ledger or existing outcome statuses.
