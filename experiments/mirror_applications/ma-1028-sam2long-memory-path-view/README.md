# MA-1028 — SAM2Long memory-path feature views

Status: SCREENING. The protocol is frozen in `PROTOCOL.json` before the MA-1028 scenes are generated.

## Hypothesis

Three SAM2Long path memories at each frame may be stored as one shared memory base plus two per-path Givens angles without losing the useful diversity of the tree. Mirror must save at least 25% complete-session bytes, remain within 1 pp J&F of independent pathways, and outperform a rank-1 additive memory control.

## Prior-art delta

PA316's official SAM2Long keeps a fixed-width tree and selects candidate branches by cumulative mask/IoU scores. This experiment factors the per-path memory feature maps while charging the unchanged path masks, scores, pointers, positional encoding, video cache, prompts, checkpoint and metadata. It does not claim to compress tree search itself.

## Methods and execution

See frozen `PROTOCOL.json`. The implementation uses official SAM2Long source commit `d70b50a7936fec55af20123530e92e794ad901a4` and official SAM2.1 Hiera-tiny weights. The session runner and exact replay verifier are under `source/`.

## Results

To be written after fresh-seed execution and verification.
