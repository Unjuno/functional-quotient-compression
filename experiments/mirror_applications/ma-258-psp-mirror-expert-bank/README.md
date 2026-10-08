# MA-258 — Parameter-superposed expert bank with Mirror unbinding

Status: **FAIL by the frozen aligned payload gate**. Fresh seeds 25831–25833 remain sealed. The unrelated regime is a clear quality boundary.

## H

One shared expert matrix plus one Mirror angle per expert should compress an aligned bank while preserving task functions; the same orbit representation should fail on unrelated experts. Required controls are fixed-sign PSP, hard tying, rank-2 SVD and independent weights.

## T

Clean rerun on two development seeds (25821/25822), each with aligned and unrelated banks of eight 16x16 linear experts. The aligned regime is constructed by rotations of one base matrix; unrelated matrices are independent at matched norm. Mirror angles are fit directly to the matrix bank for 2,000 updates. Every method is serialized to FP16 ZIP/NPY and scored after reload on 512 random inputs per expert. This is oracle weight-space compression, not trained MoE evidence.

## D

**FAIL by the preregistered total-byte threshold.** In both aligned worlds, Mirror reconstruction had zero measured normalized MSE and used 1,327 B versus 2,592 B for rank-2 SVD (0.512x); the frozen cap was 0.50x. Independent experts used 4,644 B. PSP used 1,582 B but had severe interference (max expert normalized MSE 7.03–9.31). In unrelated worlds, Mirror max expert normalized MSE was 0.87–0.89; SVD was 0.83–0.86. Fresh was not opened after the byte gate miss.

## C

The aligned bank is generated exactly from the Mirror rotation orbit, so its zero error is a favorable construction. Ordinary rank-2 SVD also reconstructs it exactly and is only slightly above the strict storage threshold. The result does not show a Mirror-specific quality advantage over this simpler shared-basis control.

## U

The aligned mechanism is not fresh-replicated because the registered development byte gate failed. The unrelated limitation has only development evidence. No learned expert training, routing, Transformer/MoE task quality or runtime inference benchmark was performed. Pre-existing untracked artifacts in this directory were excluded.

## Fact / interpretation / hypothesis

- **Fact:** All 20 clean development payloads replay; aligned Mirror quality is exact at measured precision, but total bytes miss the strict rank-2 cap. Unrelated Mirror reconstruction is poor.
- **Interpretation:** A low-description orbit can represent deliberately orbit-aligned experts compactly, but cannot represent unrelated expert functions; rank-2 SVD captures the same aligned bank with a small byte penalty.
- **Hypothesis:** For trained expert banks with genuine shared rotational structure, Mirror codes could provide a useful storage point; this remains unestablished without a compliant frozen gate and fresh replication.
