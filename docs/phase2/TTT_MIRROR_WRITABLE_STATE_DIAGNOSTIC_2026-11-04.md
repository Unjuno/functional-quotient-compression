# TTT Mirror writable-state diagnostic — 2026-11-04

## Scope

MA-651 tests a shared rank-4 dictionary with a per-stream mode code for a synthetic online linear TTT state. MA-652 adds orthogonal private writable residual ranks 0/2/4/8.

## Facts

- MA-651: shared dictionary state is about 41.5–41.7% of full TTT-Linear state. It preserves online quality on aligned synthetic streams, but private rank-4 state is slightly smaller in actual serialized bytes. Mirror mode codes are exactly ordinary dictionary IDs, with identical state and output.
- MA-652: shared+private residual state is 22.4%, 34.1%, 45.6%, and 68.8% of full TTT state for residual rank 0, 2, 4, and 8. Shared-only tail MSE grows to 1.5x/2.3x/3.6x full as private target rank increases; matching residual ranks restore quality. Shared dictionary bytes are modestly below independent private bases at ranks 2/4/8, but native mode IDs exactly match Mirror.
- Both experiments are synthetic, use two development seeds, passed tests/replay, and kept fresh seeds sealed after exact native-control alias.

## Interpretation

Low-rank writable TTT state can reduce storage for the tested aligned task. Higher private rank restores sequence-specific functions at a clear byte and compute cost. The address used to select a shared mode is an ordinary ID; the two tests do not show a Mirror-specific functional gain.

## Queue decision

Pause further variants of the same TTT shared-basis + mode-ID writable-state design until a proposal tests a Mirror coordinate that beats native IDs, private low-rank state, or another matched control. Continue other test-time-memory candidates, starting with MA-653 Titans long-term memory; this pause does not apply to Titans or other memory architectures.

## Unresolved

The tasks are supervised diagonal/linear synthetic streams, not self-supervised TTT or language modeling. Natural TTT hidden-state distributions, retrieval interactions and learned initialization are untested.
