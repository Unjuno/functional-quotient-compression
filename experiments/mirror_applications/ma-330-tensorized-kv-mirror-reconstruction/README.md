# MA-330 — Tensorized KV cache/view reconstruction

Status: PASS (synthetic mechanism gate; no natural-language claim)
Evidence lane: MECHANISM / STORAGE / RUNTIME
Base commit: `28194ea` (`research/mirror-application-worker-ready-20261007`)

## H — hypothesis

On a deliberately aligned synthetic cache, compact shared sequence/channel factors plus charged role-specific Mirror coordinates can reconstruct useful per-role K/V tensors with lower serialized bytes than a strong PA35-style shared matrix-bank control, while reporting the cost to reconstruct and consume each logical cache view.

## Mirror insertion

> **Mirror insertion:** this experiment adds a per-role orthogonal channel coordinate `m` to the shared base K/V tensor reconstruction so that role-specific logical KV caches can be expressed without storing a separate factorization for every aligned role.

- Native method: low-rank/Tucker compression of per-role KV cache tensors.
- Exact insertion: undo a known per-role channel rotation, pool aligned K/V tensors into one shared base, compress that base, then reapply each paid role angle during reconstruction.
- Persistent `m`: one float32 angle per aligned role, serialized and charged.
- Simplest control: PA35-style shared sequence basis with role-specific coefficient matrices; also independent per-role low-rank Tucker and hard sharing.
- No coordinate is learned in this screen; the angles are task metadata generated with the teacher and fully charged. No claim of learned address efficiency.

## Physical-to-logical claim

- Physical object: compressed shared K/V tensor factors for a sequence of cache positions and channels.
- View: per-role channel rotation; two unrelated roles use private factors.
- Claimed multiplicity: six aligned roles reconstructed from one shared factor pair, with two unrelated roles as a private-state boundary.
- Why it may save: avoids six duplicate role factor pairs.
- Why it may fail: the shared-angle code can be redundant with ordinary coefficient matrices, and reconstruction adds decode compute and workspace.

## Prior-art delta

PA08 MLKV shares KV heads across layers. PA35 already uses a shared matrix bank with per-layer coefficients. MA-061 found one KV head plus Mirror views inferior to MQA on its language/cache screen; MA-245 found aligned cache views promising but slower and missed its strict model-payload gate. This experiment isolates *compression of the sequence×channel cache tensor itself*, compares against a PA35 shared sequence-basis coefficient bank, and accounts for reconstruction work and materialized cache state. It does not repeat only role-level cache sharing.

## Controls

1. MHA/raw per-role cache payload.
2. Hard-tied shared cache.
3. Independent per-role rank-r SVD factors.
4. PA35-style shared sequence basis with role-specific coefficients.
5. Mirror shared-base factors plus six charged angles and two private role factors.

## Gates

### PASS
On all three fresh worlds, Mirror attention-output normalized MSE ≤ 1e-4, at least 10% fewer serialized bytes than the PA35 matrix-bank control at matched rank/quality, and decode MAC proxy ≤ 2× the matrix-bank control.

### FAIL
The fresh quality gate fails, or the Mirror has <10% byte advantage over the best non-Mirror shared-factor control at matched quality, or decode compute exceeds 2×.

### NOT ESTABLISHED
Serialization or replay fails, or results depend on a protocol deviation. This is a synthetic mechanism screen only; even a PASS does not establish language-model cache compression.

## Tuning boundary

Development worlds 33001 and 33002 select one rank from {2,4,6,8} using mean attention-output normalized MSE. Fresh worlds 33031–33033 use the frozen selected rank and are not opened before protocol/source freeze.

## Storage and compute

The deterministic serialized payload includes float32 factors, per-role angles, role/task indices, method ID, shape/dtype metadata and header. The actual byte length is authoritative. Expanded logical K/V cache bytes, peak streaming reconstruction workspace, decode MAC proxy, attention MAC proxy, wall time, and modeled read/write bytes are separate metrics. The cache tensor is reconstructed one role at a time; source payload is reloaded from bytes before scoring.

No optimizer updates are used. This is factor fitting/reconstruction from teacher tensors; there are no training examples or language tokens.

## H / T / D / C / U

**H — Hypothesis:** supplied low-description rotations over one shared compressed KV base can recover aligned logical cache roles with fewer serialized bytes than a native PA35-style coefficient bank at matched output quality, with reconstruction cost within 2x.

**T — Trial:** NumPy CPU mechanism screen; four layers × two KV roles; sequence 16, dimension 8; six aligned roles and two unrelated roles. Rank 4 was selected on development seeds 33001–33002. Three frozen fresh seeds 33031–33033 compared raw MHA, hard sharing, independent rank-4 SVD, PA35 shared sequence basis with private factors, and Mirror shared base plus charged angles/private factors. No optimizer updates.

**D — Decision:** PASS for the preregistered synthetic mechanism gate; registry status PROMISING. Mirror used 3,137 serialized bytes vs PA35 4,802 at matched near-zero attention-output MSE, while using 2× the decode MAC proxy.

**C — Strongest counter-hypothesis:** the teacher was created by the same shared-base-plus-rotation construction tested by Mirror, and angle values were supplied as metadata. Therefore the result shows coding efficiency for known aligned structure; it does not show that trained models naturally contain this structure or can learn the code efficiently.

**U — Unconfirmed:** natural Transformer KV alignment, learned address extraction, autoregressive NLL, cache bandwidth, GPU/fused decode runtime, and any fixed-byte capacity frontier.

## Decision

FACT: At frozen rank 4, 3/3 fresh seeds had normalized attention-output MSE < 2e-14 for Mirror and PA35. Actual payload was 3,137 B Mirror vs 4,802 B PA35 (34.7% fewer); all methods materialized 8,192 B logical K/V state. Mirror reconstruction used 16,384 MACs vs 8,192 for PA35, and mean CPU reconstruction+attention time was 0.255 ms vs 0.205 ms (small, noisy NumPy benchmark).
INTERPRETATION: The predeclared synthetic mechanism gate passed on payload savings at matched quality, with a 2x decode MAC cost and no expanded-cache storage reduction.
HYPOTHESIS: A low-description shared-coordinate KV tensor representation may improve the payload frontier for naturally aligned layer/head caches; this remains untested.
BOUNDARY: The teacher was generated from the exact shared-base-plus-rotation form; angles were supplied and charged rather than learned. No pretrained model, language NLL, bandwidth, GPU kernel, or natural cache alignment was tested.
