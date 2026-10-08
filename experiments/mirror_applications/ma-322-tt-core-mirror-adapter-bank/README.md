# MA-322 — Shared TT adapter cores with Mirror phase views

Status: FAIL on the frozen storage gate. Dedicated branch: `research/ma-322-tt-core-mirror-adapter-bank-20261008`.

## H — hypothesis

One phase per task will modulate a selected shared TT core and save at least 10% total bytes against free two-coefficient core modulation while preserving held-out task function quality. An aligned and a mixed aligned/private bank separate where private TT factors become necessary.

## Mirror insertion and controls

The adapter is a 64x64 matrix represented as an order-4 8x8x8x8 TT tensor with ranks 4. Three TT cores are shared; the selected second core is modulated by a two-direction orbit. Mirror stores a phase per task, while the direct control stores two free coefficients. Other controls are hard tie, independent LoRETTA-style TT factors and independent full matrices. In the mixed bank, both shared methods use the same validation-selected private TT-factor fallback for unrelated tasks.

## Frozen protocol

See `PROTOCOL.json`. Each task has 64 support, 32 validation and 64 test vectors. Development seeds 32201/32202; fresh seeds 32211/32212/32213. No optimizer updates. Actual serialized ZIP/NPY payload includes all shared/private cores, orbit directions, codes, IDs, metadata and headers.

## Development facts

Both development seeds showed 0 private fallbacks in the aligned bank and exactly 32 in the mixed bank for each shared method. Mirror reconstructed the aligned functions under the quality threshold. Its payload was 5,048 bytes versus 5,074 bytes for the direct coefficient control (0.51% fewer bytes), far below the frozen 10% savings gate. The phase-search operation proxy was 1,610,612,736 versus 1,048,576 for direct fitting.

## H/T/D/C/U report

**H — Hypothesis.** A phase per task on a shared TT core would save at least 10% total payload against two free coefficients per task, with held-out normalized function MSE at most 1e-4, in aligned and mixed banks.

**T — Trial.** Order-4 8x8x8x8 TT, ranks 4, 128 tasks per bank; aligned tasks use a planted phase orbit, mixed tasks have 96 orbit and 32 unrelated functions. 64 support / 32 validation / 64 test vectors per task, no optimizer updates; dev seeds 32201/32202, fresh seeds 32211/32212/32213. Controls were hard tying, independent TT factors (LoRETTA-style), free two-coefficient shared core with identical private fallback, and independent full FP16 matrices.

**D — Decision: FAIL for the preregistered storage promotion gate.** On aligned fresh banks Mirror was 5,048B versus 5,074B direct (0.51% smaller) in 3/3, with max test nMSE 5.25e-7–6.44e-7 and no private fallback. On mixed banks Mirror was 25,464B versus 25,426B direct (0.15% larger) in 3/3; both stored 32 unrelated tasks privately, and max test nMSE was at most 6.44e-7. The aligned savings miss the required 10%; mixed bytes are slightly worse. Mirror fit proxy was 1.611B operations vs 1.049M for direct fitting (~1,536x). Thus the strict gate failed despite quality passing.

**C — Strongest counter-hypothesis.** The two-dimensional free coefficient pair represents this phase orbit at effectively the same serialized size and is far cheaper to fit; ZIP/NPY metadata overhead erases the nominal one-scalar code saving, and the phase search adds substantial work.

**U — Unverified.** This is a planted post-fit synthetic matrix-function screen with oracle task IDs, not a learned adapter bank, pretrained model, real LoRETTA reproduction, routing experiment or capacity test. No GPU runtime frontier or trained quality is established.

### Evidence categories

- **Fact:** fresh payload sizes, quality values, task allocations and operation proxies are in `FRESH_RESULTS.csv` and `ALLOCATION_EVENTS.csv`. Thirty serialized payloads were hash/length checked and reloaded; 30 summary rows and 1,536 allocation events replayed exactly across all five methods, two banks and three fresh seeds.
- **Interpretation:** TT-core phase coordinates meet function quality on the deliberately aligned orbit, but do not provide a material end-to-end storage improvement over free coefficients. In the mixed bank, private factors for unrelated functions erase the tiny aligned saving.
- **Hypothesis:** a Mirror code with fewer charged archive objects, or a much larger aligned task bank, might improve the total-byte ratio. It would still need a lower-cost fit procedure and a native learned-task validation before a systems claim.
