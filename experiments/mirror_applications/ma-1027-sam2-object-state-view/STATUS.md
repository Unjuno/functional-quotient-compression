# MA-1027 status

Stage: COMPLETE — FAIL. Frozen protocol preceded scene generation and scoring.
Branch: `research/ma-1027-sam2-object-state-view-20261115`.
Base evidence: MA-1026 provenance commit `79944e175`.
Official SAM2.1 Hiera-tiny source commit, checkpoint, and config hashes are recorded in `runs/development.json`; model assets are cached outside Git. Full four-method forward replay is exact. Six tests pass.

H: Two object-specific Givens angles should recover each object’s SAM2 memory from one shared frame memory with at least 25% smaller session payload, <=1 pp J&F loss to independent memory, and >=1 pp advantage over rank-1 additive.

T: Official SAM2.1 Hiera-tiny, CPU, 256 px, four prompted objects, one deterministic five-frame synthetic crossing/occlusion clip; independent, hard-shared, Givens Mirror, and rank-1 additive controls.

D: FAIL. Mirror J&F is 0.0304 vs 0.8761 independent, 0.0302 hard-shared, and 0.3688 rank-1 additive. Object-state bytes fall 23.84% and full-session bytes only 0.30%; the 25% full-session gate and both quality gates fail.

C: The memory feature bank may not be compressible with two channel-plane rotations: the rank-1 additive control reconstructs substantially better, while prompt masks, output masks, pointers, scores, model weights and common frame cache remain charged/invariant.

U: Natural videos, longer clips, other object counts, other SAM2 sizes, CUDA throughput, and trained/learned coordinates remain untested. This one synthetic clip is a mechanism screen only.
