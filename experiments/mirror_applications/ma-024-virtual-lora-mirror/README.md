# MA-024 — one LoRA to many virtual LoRAs

Status: **FAIL at development screen**. Fresh worlds remained sealed because the Mirror coordinate did not show a useful frontier against generic shared-basis LoRA generation.
Evidence lane: MECHANISM / STORAGE / RUNTIME.
Branch: `research/ma-024-virtual-lora-mirror-20261007`.
Base commit: full SHA in `PROTOCOL.json`.

## H — Hypothesis

One shared rank-2 LoRA basis plus per-task rank-space Mirror rotations can represent eight adapter functions at lower serialized bytes than independent per-task LoRA. Generic shared-basis coefficients and a small task hypernetwork are the key controls.

## T — Test

Synthetic frozen-feature 16D-to-12D adapter regression with eight oracle task IDs. Compared independent rank-2 LoRA bank, one shared LoRA, generic two-basis learned coefficients, task-embedding hypernetwork, and Mirror rank rotations. Teacher modes were generated from a shared rotated LoRA and eight independent rank-2 LoRAs. Development world 240000 selected common LR 0.01 from {0.003, 0.01}, with 1,200 updates. Fresh worlds were not opened after development indicated a weak Mirror-specific frontier.

## D — Decision: FAIL (development)

### Fact

- On the aligned teacher at LR 0.01: Mirror MSE 7.76e-10; independent LoRA bank 1.66e-10; generic shared basis 2.37e-11. At LR 0.003 Mirror MSE was 6.45e-7 vs full bank 2.15e-6, but generic control was 6.59e-11.
- Serialized payload: Mirror 2,401B, generic shared basis 2,657B, full LoRA bank 3,749B. Mirror saved 36.0% vs full bank but only 9.6% vs generic shared-basis coefficients, below the preregistered 10% Mirror-specific byte advantage.
- Mirror and generic basis used the same compute proxy (8.60M), twice the full per-task LoRA proxy (4.30M). Development inference throughput was 0.386M Mirror vs 0.576M generic vs 0.760M full LoRA examples/s.
- In independent-LoRA mode, Mirror MSE was 1.76 vs full-bank 1.41e-10; the two-basis generic control was also far from full bank at 1.50.
- Twenty development rows replayed; payload bytes exact, max MSE delta 4.5e-10, worst-task delta 4.5e-10, R² delta 4.9e-9. Tests: 3 passed. Fresh worlds 240001–240003 remained unopened.

### Interpretation

Sharing a LoRA basis reduces bytes versus a full bank, but one-angle Mirror rotations did not outperform ordinary low-rank coefficient generation. Generic basis fit the aligned teacher more accurately at only 256B extra, while Mirror had the same compute proxy and lower measured throughput. This screen does not support a Mirror-specific advantage.

### Strongest counter-hypothesis

A different angle initialization or per-method LR may improve optimization. Only a common LR selected over two candidates was tested; retuning now would require a new registered development amendment before any fresh split.

### U — Unconfirmed

Natural language, nonlinear adapters, learned task inference, per-method optimization, larger adapter rank, and near-convergence fixed-byte frontiers remain untested.

## Fact / interpretation / hypothesis

- **Fact:** full LoRA bank compresses poorly; shared-basis methods reduce bytes; generic coefficient generation beat Mirror quality in development at comparable payload.
- **Interpretation:** this Mirror coordinate is a constrained form of shared-basis LoRA generation without a useful observed quality/storage/compute advantage.
- **Hypothesis:** a different structured coordinate may help when the task family has additional symmetry; not tested here.
