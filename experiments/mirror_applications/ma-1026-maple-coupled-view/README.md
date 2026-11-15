# MA-1026 — MaPLe coupled-modality prompt views

Status: FAIL (bounded CPU screen). See `PROTOCOL.json` for the frozen paired-branch code, controls, data split and gates.

The model adapts the official MaPLe coupled design: two shallow text tokens project into the visual prompt stream, and one deeper text prompt has a corresponding visual projection. Since only two context positions are used, each Mirror coordinate rotates the token pair in its assigned modality. The learned visual/text coordinates remain separate. The additive control applies the same two scalar domain codes through learned rank-1 text and visual bases.

This is a bounded CPU screen using ViT-B/32 and a two-level prompt depth, rather than the official ViT-B/16, depth-9, five-epoch benchmark configuration. The official implementation commit is recorded in the protocol. It must not be read as a reproduction of published MaPLe numbers.

## Result

Mirror learned state uses 3,161,136 B versus 12,641,981 B for independent per-domain MaPLe, a 75.0% reduction, and matches its novel-class accuracy. But it is 1.28 pp below the rank-1 additive control and ordinary shared MaPLe in H-mean, while the held-out base-domain cells tie all three shared-state methods. This is a FAIL for Mirror-specific value. The complete record and limitations are in `RESULTS.md`.
