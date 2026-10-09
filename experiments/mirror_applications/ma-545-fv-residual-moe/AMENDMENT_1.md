# MA-545 protocol amendment 1 — deterministic RNG seed

Date: 2026-10-09 UTC

## Reason

The frozen protocol specified optimizer, updates and data split, but omitted router initialization RNG. Initial development runs therefore used PyTorch's ambient initialization state. Exact replay requires defining this seed.

## Change

Set NumPy and PyTorch CPU RNG seeds to the world seed before extraction and router initialization. All other data, architecture, update count, optimizer, thresholds, prompt format, controls and metrics remain unchanged. The five-thread CPU setting remains fixed.

## Data handling

Initial development artifacts are preserved under `results/pre_amendment_1/`; they are exploratory and are not substituted for the amended primary result. Fresh seeds remain unopened. Re-run both development seeds under this amendment before deciding whether to open fresh seeds.
