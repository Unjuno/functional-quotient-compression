# MA-807 — Factorized embodiment × task policy codes

Status: **FAIL for Mirror-specific attribution and the 25% storage gate**

## Prior-art boundary

PA202 studies invertible adapters in one-step flow-matching policies. This screen isolates embodiment/task composition in an 8D linear invertible action map. No multi-robot data, simulator, or hardware is used.

## Fact

Seed 80701; three embodiments × three tasks; five checkerboard pairs held out; 256 training examples per observed pair (1,024 total) and 512 test examples/pair; zero optimizer updates. Independent pair matrices use 1,867 B. The factorized shared base plus embodiment/task Givens codes uses 1,534 B, reaches heldout nRMSE 8.27e-8, success 1.0, and orthogonality error 3.90e-7. This is 17.8% fewer bytes, below the 25% threshold. Embodiment-only and task-only controls have heldout errors .143/.154 and success .065/.017. Native additive Givens composition has the exact same payload/hash and action outputs. Factorized Givens uses 76 MAC/action versus 64 for independent dense matrices.

## Interpretation

Separating the embodiment and task factors recovers the heldout cross-pairs in this designed system. However, the byte savings miss the registered gate, and ordinary additive Givens angles exactly express the factorization. The View also costs more active MACs than independent matrix multiplication. No Mirror-specific benefit or cross-embodiment robotic claim is established.

## H / T / D / C / U

- **H:** Separate embodiment and task coordinates in an invertible latent action space recover heldout pairings with one-step success and at least 25% fewer complete bytes than independent policies.
- **T:** Seed 80701; 8D orthogonal base; 3×3 embodiment/task grid; checkerboard five heldout pairs; 256 training/512 test examples per pair. Compare independent pair policies, embodiment-only, task-only, shared base, factorized View and native additive Givens. Measure complete NPZ bytes/hash, heldout/all-pair action nRMSE, success, orthogonality, MACs, examples, CPU one-step timing and exact replay.
- **D:** **FAIL** Mirror-specific/storage gate. Heldout actions are recovered and success is 1.0, but storage savings are 17.8%, native Givens aliases exactly, and active MACs are 18.75% above dense independent policies.
- **C:** The factorization is standard additive composition in the Givens rotation group; no Mirror-specific coordinate is needed.
- **U:** Real robots, unseen physical embodiments, nonlinear action dynamics, learned task/embodiment identification, flow-policy training, safety and hardware latency.

Fresh pairings stayed sealed after the byte gate miss and exact native alias. This is synthetic policy evidence only.
