# MA-322 — Shared TT adapter cores with Mirror phase views

Status: development complete; frozen-source fresh runs pending. Dedicated branch: `research/ma-322-tt-core-mirror-adapter-bank-20261008`.

## H — hypothesis

One phase per task will modulate a selected shared TT core and save at least 10% total bytes against free two-coefficient core modulation while preserving held-out task function quality. An aligned and a mixed aligned/private bank separate where private TT factors become necessary.

## Mirror insertion and controls

The adapter is a 64x64 matrix represented as an order-4 8x8x8x8 TT tensor with ranks 4. Three TT cores are shared; the selected second core is modulated by a two-direction orbit. Mirror stores a phase per task, while the direct control stores two free coefficients. Other controls are hard tie, independent LoRETTA-style TT factors and independent full matrices. In the mixed bank, both shared methods use the same validation-selected private TT-factor fallback for unrelated tasks.

## Frozen protocol

See `PROTOCOL.json`. Each task has 64 support, 32 validation and 64 test vectors. Development seeds 32201/32202; fresh seeds 32211/32212/32213. No optimizer updates. Actual serialized ZIP/NPY payload includes all shared/private cores, orbit directions, codes, IDs, metadata and headers.

## Development facts

Both development seeds showed 0 private fallbacks in the aligned bank and exactly 32 in the mixed bank for each shared method. Mirror reconstructed the aligned functions under the quality threshold. Its payload was 5,048 bytes versus 5,074 bytes for the direct coefficient control (0.51% fewer bytes), far below the frozen 10% savings gate. The phase-search operation proxy was 1,610,612,736 versus 1,048,576 for direct fitting. These are development observations only; frozen settings proceed to fresh evaluation.
