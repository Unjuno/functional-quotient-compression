# MA-451 — PathNet path with Mirror module-role views

Status: SCREENING  
Evidence lane: MECHANISM/STORAGE/RUNTIME  
Branch: `research/ma-451-pathnet-mirror-role-20261008`  
Base commit: `1569029`  
Prior art: PA80 (PathNet)

## Hypothesis

H: Two adapted Mirror angles can let one reused PathNet path serve two distinct role orientations, improving held-out query quality over path-only and rank-2 control at a useful actual-byte/runtime point. A direct native Givens implementation can falsify Mirror-specific attribution.

**Mirror insertion:** this experiment adds two support-adapted role angles to the output of a selected PathNet module path so the shared path can implement two logical role functions without allocating duplicate modules.

## Task and controls

Two module choices per layer make four PathNet paths. For each path, two distinct angle-role tasks form an eight-task bank. Models meta-train shared module weights on synthetic episodes and adapt role coordinates on support examples. PathNet path-only (no role coordinate) is mandatory; additional controls are rank-2 additive residual per path, native Givens conditioning, and independent support-fit vectors. The task is a small linear regression mechanism screen, not a reproduction of the full PathNet paper.

Two development seeds (45101, 45102), 12 support examples, 40 query examples and four inner updates are frozen in `PROTOCOL.json`. Fresh seeds 45111–45113 stay sealed unless every development gate passes. Every path index, module, view code, residual basis and task vector is charged in an actual uncompressed `.npz` payload.

## Results

Facts, interpretation and hypothesis will follow the frozen development run and payload replay verification.
