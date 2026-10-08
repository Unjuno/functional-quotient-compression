# MA-303 — Factorized layer-mask Mirror codes

Status: amendment A1 runner frozen; fresh evaluation pending. Dedicated branch: `research/ma-303-factorized-layer-mask-mirror-20261008`.

## H — hypothesis

A factorized Mirror address for task x layer mask combinations will recover held-out factor pairs on a fixed random two-layer ReLU network and provide a material actual-byte improvement over Piggyback masks and a direct shared-factor coefficient control. Unrelated masks should require private state.

## Mirror insertion

Native method is Piggyback-style binary task masks over a fixed random backbone. Mirror inserts a rank-2 factor interaction into the edge mask, generated from task and layer coordinates on a shared basis. The cheapest ordinary control is a direct real-valued rank-2 task x layer coefficient bank; independent masks are the upper control.

## Frozen protocol

See `PROTOCOL.json`. Six task factors and six layer factors are observed during development/training. Four combinations from held-out factor IDs are tested in fresh worlds. Each pair has 256 support, 128 validation and 256 test examples. Seeds: development 30301/30302; fresh 30311/30312/30313. No optimizer updates. Actual fixed-timestamp ZIP/NPY bytes are authoritative.

After development, implementation audit found that the initial runner did not apply the declared validation fallback. Amendment A1 added the fallback and explicit split accounting without changing seeds or gates. Initial development JSON is preserved separately as pre-amendment and excluded from the corrected result table.

## H/T/D/C/U report

**H — Hypothesis.** Rank-2 task and layer Mirror coordinates on a fixed random ReLU backbone would compose held-out factor pairs at test nMSE <=0.05, saving >=20% actual bytes against Piggyback masks and >=10% against direct factorized coefficients, with no extra private fallbacks.

**T — Trial.** Two 64x64 ReLU layers, 25% binary edge masks, 8 task and 8 layer factors, and 4 held-out pairs (6,6), (6,7), (7,6), (7,7). Controls were no-mask shared backbone, Piggyback masks on observed 6x6 pairs, direct Cartesian rank-2 factor codes, phase-based Mirror coordinates, and independent masks for all pairs. Corrected dev seeds 30301/30302; fresh seeds 30311/30312/30313; 256 support, 128 validation and 256 test inputs per pair; zero optimizer updates and oracle planted factor coordinates. Amendment A1 froze before fresh access.

**D — FAIL for the Mirror-specific promotion gate.** Fresh Mirror payload was 34,892B (3/3) versus 34,920B direct factorized: 28B or 0.080% smaller, far below the 10% gate. It was 51.0% smaller than Piggyback masks (71,243B) and 65.1% smaller than independent binary masks (99,915B), but those controls do not compose held-out factors: held-out max nMSE was 15.2–16.5. Mirror and direct-factorized had max nMSE 0.0 over all 64 pairs, including four held-out pairs, with no private fallbacks. The shared no-mask model also failed held-out quality. Mirror/ direct CPU throughput was similar (Mirror/direct 1.01–1.16x across three seeds); both were slower than pre-stored Piggyback/independent masks. This eager NumPy measurement is a harness result, not a deployment speed claim.

**C — Strongest counter-hypothesis.** The gain over binary mask storage comes from factorized task/layer codes, not a Mirror-specific effect. Ordinary normalized Cartesian rank-2 codes reconstructed the same masks and quality within 28 serialized bytes; both methods store the same shared backbone and basis, so archive overhead dominates.

**U — Unverified.** Factor coordinates are planted/oracle and stored for all eight task and layer identities, including held-out IDs; this is a post-fit composition screen, not learned held-out generalization or continual learning. No optimizer, natural tasks, GPU kernel, or trained-model capacity is tested. The observed 64 functions do not imply 64 independent capabilities.

### Evidence categories

- **Fact:** 15 fresh payloads were reloaded and matched byte lengths and SHA-256; rerunning all three fresh seeds exactly reproduced 15 summary metrics and hashes (max difference 0). Four tests pass.
- **Interpretation:** factorized mask addresses can replace per-pair binary masks when task and layer coordinates are already available, but the Mirror phase parameterization adds no material advantage over ordinary direct factorized codes.
- **Hypothesis:** actual learned factor codes might change the storage/quality frontier, but require a separately frozen training experiment with whole task identities held out.
