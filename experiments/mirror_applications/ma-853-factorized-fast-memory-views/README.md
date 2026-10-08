# MA-853 — Factorized key-role × value-role Mirror fast memory

Status: FAIL for Mirror-specific advantage
Evidence lane: MECHANISM / STORAGE / COMPUTE / RUNTIME
Base commit: 28194ea (research/mirror-application-worker-ready-20261007)

## H — hypothesis

A shared key-role and value/function-role code bank can compose useful associative mappings for held-out role pairs, using less writable state than storing every pair independently, while a standard Hadamard binding control determines whether the gain is Mirror-specific.

## Mirror insertion

> **Mirror insertion:** this experiment adds a pair address m=(key-role i, value/function-role j) to a shared fast-memory codebook so that a logical associative mapping can be reconstructed from role factors rather than written as a separate pair-specific matrix column.

The physical objects are the role codebooks and DeltaNet-style writable matrix state. Each codebook, pair address, write state and metadata object is charged. No claim treats 36 combinations as 36 independent learned functions.

## Prior-art delta

PA230 establishes DeltaNet's writable matrix state and delta-rule update; PA11 requires MAP/Hadamard binding as a fast VSA control. MA-853 tests whether role-factorized Views preserve recall for pairings withheld from the online write stream, and compares against both controls and a full pair table. This is a synthetic associative-memory mechanism screen, not a full DeltaNet or Transformer reproduction.

## Task and controls

Six key roles × six value/function roles yield 36 query pairs. A teacher maps each pair to the Hadamard product of its 16-D role vectors. Thirty pairs are written to the online state; six fixed pairs are held out. Fresh worlds use new role vectors. The DeltaNet control stores a D×36 fast matrix and applies one delta-rule write per observed pair. Mirror and VSA/Hadamard read the same two shared role-code banks. The independent table stores all 36 outputs.

Metrics: normalized MSE on seen and held-out pairs, exact nearest-code retrieval on held-out pairs, actual serialized state bytes, write/update MACs, query MACs, update time, and recall wall time. Role vectors are supplied by the synthetic teacher and charged; they are not learned.

## Gates

PASS for the broad mechanism requires 3/3 fresh seeds with held-out normalized MSE <=1e-6, >=10% fewer state bytes than the full pair table, and write/query MACs reported separately.

Mirror-specific pass additionally requires >=10% fewer bytes than the Hadamard VSA control at matched held-out recall. If Hadamard matches, the Mirror-specific claim fails even if factorization works.

## Selection

Random draw 10 selected MA-853 from 453 P0/UNTESTED candidates at index 345. Pool snapshot and hash are in source/selection_pool.csv and PROTOCOL.json. No MA-853 branch was found.

## H / T / D / C / U

H: see hypothesis above.
T: pending frozen run; see PROTOCOL.json.
D: FAIL for Mirror-specific benefit. The factorized code passes the broad held-out recall/storage gate, but Hadamard binding matches quality and uses 7 B fewer.
C: Hadamard binding is the same elementwise composition and may match Mirror exactly; DeltaNet may retain more useful state for seen pairs.
U: learned role vectors, noisy keys, long-sequence stability, full DeltaNet gating, and natural-language memory use.

## H / T / D / C / U

**H:** Shared key-role and value/function-role factors may recover unseen pairings with less writable state than a full pair table.

**T:** 6×6 pairs, 16-D role vectors, 30 DeltaNet-style writes and six held-out pairs; fresh seeds 85311–85313. Compared delta rule, Mirror product, Hadamard VSA and full pair table.

**D:** Broad factorized-memory gate passes; Mirror-specific gate FAILs against Hadamard binding.

**C:** Hadamard binding is algebraically identical to the Mirror product and uses 7 fewer serialized bytes.

**U:** Learned roles, noisy/overlapping keys, dynamic code updates, longer sequence use, and real DeltaNet/Transformer behavior.

## Fact / Interpretation / Hypothesis

FACT: On 3/3 fresh seeds, factorized Mirror and Hadamard VSA achieve zero held-out MSE and exact recall 1.0. Mirror payload is 1,207 B vs VSA 1,200 B and full pair table 2,667 B. Delta-rule state writes 30 pairs but held-out MSE averages 0.866 and exact nearest recall 0.056.
INTERPRETATION: Static role factors compose the held-out pairs and save 54.8% of bytes against the full pair table; standard Hadamard binding is the same mechanism and slightly smaller, so the Mirror-specific gate fails. These 36 combinations are not 36 independent functions.
HYPOTHESIS: Factorized compositional state may help where role binding matches the task structure, while writable DeltaNet state remains useful for observed arbitrary associations.
BOUNDARY: Synthetic role-pair associative recall only; role vectors supplied by teacher and not learned.
