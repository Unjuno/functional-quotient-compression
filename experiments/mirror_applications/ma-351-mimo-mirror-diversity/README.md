# MA-351 — MIMO with Mirror member views

Status: **FAIL at development gate; fresh sealed**
Dedicated branch: `research/ma-351-mimo-mirror-diversity-20261008`
Base commit: `52d734a`
Prior art: PA42, MIMO implicit ensemble subnetworks.

## H — Hypothesis

On a controlled nonlinear classification problem, a shared MIMO trunk with one learned low-description Mirror member coordinate can lower member prediction correlation and improve ensemble NLL over ordinary single-forward MIMO, while using fewer actual serialized bytes than member-specific readout heads.

Falsification: Mirror does not improve diversity/NLL at comparable payload, or ordinary low-dimensional member coefficients provide the same freedom.

## Mirror insertion

> **Mirror insertion:** this experiment adds `m` to the MIMO output readout interface so that four member classifiers can be expressed from a shared trunk without storing a separate full output head for every member.

- Native method: shared hidden trunk and four MIMO readout heads, jointly supervised in one shared-trunk forward.
- Exact insertion: `head_k = V0 + m_k * V1`, with learned scalar per member.
- Persistent `m`: one scalar per member.
- Cheapest control: same two readout basis matrices and direct scalar member coefficients. Rank-2 direct coefficients are an additional capacity control.
- Physical object: one shared two-layer MLP trunk and readout state.
- Logical multiplicity: four supervised member functions.

## Prior-art delta and controls

PA42 already establishes multiple MIMO subnetworks in one forward. Required controls: four independent MLPs; native shared-trunk MIMO; rank-1 shared readout with direct coefficients; the same rank-1 readout encoded as Mirror scalar coordinates; rank-2 direct readout. The Mirror/direct comparison isolates parameterization only; it is not enough that a model has multiple outputs.

## T — Frozen/amended test

Synthetic 2D four-class nonlinear angular-sector task, 8,192 train and 4,096 test examples per world; four members; MLP widths [32,32]; 2,000 AdamW updates at 1e-3, batch 128. Development seeds 35101/35102. Fresh seeds 35111/35112/35113 remained sealed.

An invalid first attempt used noisy angular labels and a malformed independent-control label index; shared methods also stayed near uniform NLL. Before any fresh access, the protocol was amended to deterministic angular sectors and shared labels across members. The fixed update budget, architectures, optimizer, storage contract, gates, and seeds did not change. Only the amended development result is reported. A second metric audit separated pooled ensemble NLL from per-member NLL; all amended runs had member accuracy >=0.994, so optimization completed, and corrected NLL is finite and low.

## D — Decision

**FAIL at the preregistered development gate; fresh seeds sealed.** On both development worlds, Mirror rank-1 and direct rank-1 readout had equal outputs (Jensen–Shannon diversity 0), correctness correlation 1.0, nearly equal payloads (6,790 vs 6,791B at most), and identical NLL to displayed precision. Thus shared rank-1 readout collapsed four logical members to the same function and Mirror added no member diversity. Native MIMO retained low correctness correlation (~0.82 / 0.73) with similar ensemble NLL (~0.021 / 0.027) and used ~8.7KB. Rank-2 direct readout also collapsed to one function at this task/optimization setting. Independent models had high accuracy and greater error diversity but used ~23.8KB.

## Fact / Interpretation / Hypothesis

**Fact:** In both development worlds, rank-1 Mirror and direct-code logits were exactly equal; their outputs had pairwise JSD 0 and correctness correlation 1.0. Actual payload differed by no more than 1B. Serialization replay was exact.

**Interpretation:** The scalar code did not produce useful ensemble diversity under this shared readout parameterization. Native MIMO already achieved decorrelated errors at a substantially smaller payload than independent models. Rank-1 factorization removed diversity and did not improve on MIMO NLL.

**Hypothesis:** A non-collinear multi-axis view or member-specific hidden modulation may preserve diversity, but would need to beat native MIMO and a direct coefficient control after charging all coordinates.

## C — Strongest counter-hypothesis

The task is simple and all models reach near-perfect accuracy, so diversity estimates have few errors to decorrelate. This may hide value on harder or noisier classification tasks. It does not explain the exact equality of Mirror and its direct scalar control.

## U — Unconfirmed

No fresh-world replication, natural dataset, language model, near-convergence capacity frontier, or calibrated ensemble study. Correctness correlation is assessed after fixed-budget training only.

## Reproduction

```bash
python experiments/mirror_applications/ma-351-mimo-mirror-diversity/source/run.py --dev-only
python -m pytest -q experiments/mirror_applications/ma-351-mimo-mirror-diversity/tests
python experiments/mirror_applications/ma-351-mimo-mirror-diversity/source/verify.py
```
