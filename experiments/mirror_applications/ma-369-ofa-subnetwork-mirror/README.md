# MA-369 — Once-for-All subnetwork Mirror correction

Status: **FAIL for Mirror-specific value** (development-only mechanism/storage screen)  
Branch: `research/ma-369-once-for-all-mirror-20261009`  
Base: `ab95e5d0`  
Prior art: PA52 Once-for-All

## H — Hypothesis

A small width-addressed Mirror gate on shared supernet channels can recover several width-specific linear target functions with lower actual payload bytes than independent subnetworks, and outperform a byte-matched ordinary coefficient control.

## Insertion and controls

The shared physical object is one maximum-width linear layer. A subnetwork selects a nested prefix. Mirror `m` is one per-width diagonal gain vector applied to that prefix. Controls are (1) raw shared prefix, (2) a direct learned per-channel coefficient vector, (3) the same coefficient vector named/serialized as Mirror, and (4) independently specialized per-width matrices. The direct coefficient control is deliberately functionally identical to the Mirror gate.

## Protocol and boundary

This is a synthetic mechanism screen, not a trained Once-for-All network. One teacher matrix generates three nested-width target matrices by truncating its input columns; per-width target gains are generated independently and must be learned from examples. Two development worlds use fixed seeds and 512 train / 512 test Gaussian examples, SGD for 300 updates. No fresh data are opened because the direct coefficient representation is algebraically identical by construction; the Mirror-specific gate is decided on development. All payload tensors and metadata are serialized in deterministic NPZ and counted as actual file bytes. Compute reports examples, updates, matrix MAC proxy and wall time.

## Gates

PASS for a useful mechanism point requires all three width targets nMSE ≤ 1e-3, at least 10% fewer payload bytes than independent specialization, and at least 10% fewer bytes than the direct coefficient representation at matched quality. FAIL if direct coefficients match Mirror within 1% bytes/quality, or target quality fails. A tied-prefix win alone is not Mirror-specific.

## Results

See `RESULTS_CORE.csv` and `STATUS.md`. No real OFA search, progressive shrinking, hardware latency, or language-model quality is claimed. Fresh worlds were not opened because the specificity gate already failed.

## H / T / D / C / U

- **H:** Width-specific Mirror gates recover quality from shared nested prefixes while using fewer actual bytes than independent subnetworks and native coefficient views.
- **T:** Synthetic nested linear supernet; development seeds 36901 and 36902; shared-prefix, direct coefficient, Mirror, and independent matrix controls.
- **D:** FAIL for Mirror-specific value. Mean nMSE was 0.035039 for direct coefficients, Mirror gates, and independent maps; shared prefix was 0.042589. Shared-once bundle bytes per world were 576 B (prefix), 1344 B (direct/Mirror), and 960 B (independent).
- **C:** A diagonal coefficient is an ordinary native parameter with the same function class and serialization cost.
- **U:** Actual OFA training, hardware-specific latency, natural tasks, and capacity frontier.
