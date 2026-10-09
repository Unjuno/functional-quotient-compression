# Mirror MA evidence integration audit — 2026-10-08

## Purpose and provenance

Integrate the completed MA evidence chain onto the **875-candidate** research baseline without replacing the historical 254-row worker registry or losing the newer Mirror parameter doctrine.

- Authoritative target: `research/mirror-application-worker-ready-20261007` at pre-integration `388e7ea3db00463cd953a968c55b621330431f83`.
- Source experiment chain: `research/ma-015-domain-factorized-experts-20261007` at `f74259cf0a2a0fa8620f57245257e5917f222ec1` (**46** independently indexed experiment directories).
- Separate KV algebra source: `research/ma-691-lazy-kv-mirror-20261007` (one independently indexed experiment).
- Physical transfer method: graft the **47 exact Git experiment-directory trees**, preserving source, tests, protocols, results and verification files; import only status values for IDs already known in the 875 registry.
- The 254-row registry from the experiment branch was **not** copied. The 875-row registry retained all existing candidate definitions, priorities, prior-art references and worker notes.
- The baseline's six pre-existing scientific statuses were preserved; **41** further status decisions and claim ledger entries were imported. Original experiment branches remain unchanged.
- The source's Family B KV diagnostic is copied as an additional evidence file. The research baseline's goal, worker doctrine, integration matrix and queues remain authoritative.

## Reconciled count

| Metric | Value |
|---|---:|
| Registered MA ideas | 875 |
| Independently verified/report-backed experiment directories | 47 |
| PROMISING | 29 |
| FAIL | 18 |
| UNTESTED | 828 |
| P0 (completed / untested) | 35 / 373 |
| P1 (completed / untested) | 12 / 352 |
| P2 (completed / untested) | 0 / 103 |
| Newly imported statuses | 41 |
| Duplicate or conflicting candidate IDs | 0 |

**Meaning of status:** PROMISING is evidence of a bounded experiment outcome, **not** ADOPTED, real-model compression, or a replicated natural-language capacity gain. Several PROMISING entries miss a stricter secondary gate, have runtime regressions, or rely on a deliberately Mirror-aligned synthetic teacher. The experiment README and verification JSON control the claim scope.

## Indexed report and verification sources

| MA ID | Status | Experiment report | Verification | Original evidence commit |
|---|---|---|---|---|
| MA-001 | PROMISING | [report](../../experiments/mirror_applications/ma-001-nonlinear-top1-expert/README.md) | [verification](../../experiments/mirror_applications/ma-001-nonlinear-top1-expert/VERIFICATION.json) | `60ca242afcce0ff78b5116ce57dbe454a9dc4629` |
| MA-002 | PROMISING | [report](../../experiments/mirror_applications/ma-002-mirror-top2-expert/README.md) | [verification](../../experiments/mirror_applications/ma-002-mirror-top2-expert/VERIFICATION.json) | `2b7f7fe736393ef3702daa7995e0951d54e4de12` |
| MA-003 | PROMISING | [report](../../experiments/mirror_applications/ma-003-mirror-topk-expert/README.md) | [verification](../../experiments/mirror_applications/ma-003-mirror-topk-expert/VERIFICATION.json) | `e50a20fe4c000ffb3113f9d3e6564b3efacd4395` |
| MA-004 | PROMISING | [report](../../experiments/mirror_applications/ma-004-soft-mirror-expert-mixture/README.md) | [verification](../../experiments/mirror_applications/ma-004-soft-mirror-expert-mixture/VERIFICATION.json) | `e8fea8ef10f876835c4683bad0203ec7fdc91e16` |
| MA-005 | PROMISING | [report](../../experiments/mirror_applications/ma-005-signed-mirror-mixture/README.md) | [verification](../../experiments/mirror_applications/ma-005-signed-mirror-mixture/VERIFICATION.json) | `de68ec4c440e54fc4870734e9bbd1fdc99515628` |
| MA-006 | PROMISING | [report](../../experiments/mirror_applications/ma-006-expert-choice-mirror/README.md) | [verification](../../experiments/mirror_applications/ma-006-expert-choice-mirror/VERIFICATION.json) | `572e245bee17ffe430863a4827baa21da809fb82` |
| MA-007 | PROMISING | [report](../../experiments/mirror_applications/ma-007-token-choice-mirror/README.md) | [verification](../../experiments/mirror_applications/ma-007-token-choice-mirror/VERIFICATION.json) | `877e22589af90afb3ddd0e7f422977034831bab7` |
| MA-008 | PROMISING | [report](../../experiments/mirror_applications/ma-008-hierarchical-mirror-moe/README.md) | [verification](../../experiments/mirror_applications/ma-008-hierarchical-mirror-moe/VERIFICATION.json) | `2497982dc0d2861cc89acc115cffb2877439c71e` |
| MA-009 | FAIL | [report](../../experiments/mirror_applications/ma-009-rare-private-mirror-expert/README.md) | [verification](../../experiments/mirror_applications/ma-009-rare-private-mirror-expert/VERIFICATION.json) | `41020ba0aa45d9337fec68f7f772d8d5076534a6` |
| MA-010 | FAIL | [report](../../experiments/mirror_applications/ma-010-sequential-mirror-experts/README.md) | [verification](../../experiments/mirror_applications/ma-010-sequential-mirror-experts/VERIFICATION.json) | `64caf2d8e5f338a53347f6b59628fbc6ad30fced` |
| MA-011 | FAIL | [report](../../experiments/mirror_applications/ma-011-product-mirror-experts/README.md) | [verification](../../experiments/mirror_applications/ma-011-product-mirror-experts/VERIFICATION.json) | `5ae711ee4e6721ef1cee6e78e666b7f760f609d9` |
| MA-012 | PROMISING | [report](../../experiments/mirror_applications/ma-012-residual-mirror-moe/README.md) | [verification](../../experiments/mirror_applications/ma-012-residual-mirror-moe/VERIFICATION.json) | `3cdff5dc56cc2c44a1b29112d823c9e13ee9744a` |
| MA-013 | FAIL | [report](../../experiments/mirror_applications/ma-013-multiplicative-mirror-moe/README.md) | [verification](../../experiments/mirror_applications/ma-013-multiplicative-mirror-moe/VERIFICATION.json) | `e0c7831f0547de1bb824cbe403f01bb9138e0b4d` |
| MA-014 | PROMISING | [report](../../experiments/mirror_applications/ma-014-layer-specific-expert-bank/README.md) | [verification](../../experiments/mirror_applications/ma-014-layer-specific-expert-bank/VERIFICATION.json) | `bf54b738a1e88abd0165f1fc889b1639b262d05b` |
| MA-015 | PROMISING | [report](../../experiments/mirror_applications/ma-015-domain-factorized-experts/README.md) | [verification](../../experiments/mirror_applications/ma-015-domain-factorized-experts/VERIFICATION.json) | `65e006d1f4b2fac64c4b89d89437e36f481f6ea9` |
| MA-019 | FAIL | [report](../../experiments/mirror_applications/ma-019-mirror-coefficient-basis/README.md) | [verification](../../experiments/mirror_applications/ma-019-mirror-coefficient-basis/VERIFICATION.json) | `b880e34edb8210ad318091be2e17e170ef09c5a0` |
| MA-024 | FAIL | [report](../../experiments/mirror_applications/ma-024-virtual-lora-mirror/README.md) | [verification](../../experiments/mirror_applications/ma-024-virtual-lora-mirror/VERIFICATION.json) | `4a41b4af333d181bd873f2f79751177f8c1c9a95` |
| MA-041 | PROMISING | [report](../../experiments/mirror_applications/ma-041-mirror-attention-heads/README.md) | [verification](../../experiments/mirror_applications/ma-041-mirror-attention-heads/VERIFICATION.json) | `359111bc9cc649fe29ad8793035c237dbf211cba` |
| MA-048 | FAIL | [report](../../experiments/mirror_applications/ma-048-physical4-logical16-attention/README.md) | [verification](../../experiments/mirror_applications/ma-048-physical4-logical16-attention/VERIFICATION.json) | `bdd1dd3002b493cf1355dcc297e9de23b5e0b532` |
| MA-061 | FAIL | [report](../../experiments/mirror_applications/ma-061-single-kv-logical-views/README.md) | [verification](../../experiments/mirror_applications/ma-061-single-kv-logical-views/VERIFICATION.json) | `90ae87a3c36d4a9cee79dca01881cd060266ab7b` |
| MA-063 | FAIL | [report](../../experiments/mirror_applications/ma-063-causal-mirror-mqa/README.md) | [verification](../../experiments/mirror_applications/ma-063-causal-mirror-mqa/VERIFICATION.json) | `b69a19315bcead3ea0d05e22d14c76a6e00c63b1` |
| MA-076 | PROMISING | [report](../../experiments/mirror_applications/ma-076-one-block-many-layers/README.md) | [verification](../../experiments/mirror_applications/ma-076-one-block-many-layers/VERIFICATION.json) | `418a888bfebfb9bf0ad38dbc1b3c97a5001ec7ce` |
| MA-079 | PROMISING | [report](../../experiments/mirror_applications/ma-079-learned-depth-address/README.md) | [verification](../../experiments/mirror_applications/ma-079-learned-depth-address/VERIFICATION.json) | `f1cf8d6a64e7f3fbee4b25bddfba71f0a8f731db` |
| MA-086 | FAIL | [report](../../experiments/mirror_applications/ma-086-depth-address/README.md) | [verification](../../experiments/mirror_applications/ma-086-depth-address/VERIFICATION.json) | `155231548d0c202ec1d23b708206452434a7c2b7` |
| MA-111 | PROMISING | [report](../../experiments/mirror_applications/ma-111-semantic-role-embedding/README.md) | [verification](../../experiments/mirror_applications/ma-111-semantic-role-embedding/VERIFICATION.json) | `865c8f1f842db934b94564ed0c1d97f688879740` |
| MA-116 | FAIL | [report](../../experiments/mirror_applications/ma-116-mirror-rope/README.md) | [verification](../../experiments/mirror_applications/ma-116-mirror-rope/VERIFICATION.json) | `b5d0b623df16f2ea3b024e92c53edb96f26ea00f` |
| MA-121 | PROMISING | [report](../../experiments/mirror_applications/ma-121-temporal-view/README.md) | [verification](../../experiments/mirror_applications/ma-121-temporal-view/VERIFICATION.json) | `0ccd5470dcac07afba3b9cf77bbfbb7cf8e82bfc` |
| MA-129 | FAIL | [report](../../experiments/mirror_applications/ma-129-temporal-view/README.md) | [verification](../../experiments/mirror_applications/ma-129-temporal-view/VERIFICATION.json) | `35cd8bb9fa638006828bdd61ae9511df11dc8af1` |
| MA-156 | PROMISING | [report](../../experiments/mirror_applications/ma-156-compression-view/README.md) | [verification](../../experiments/mirror_applications/ma-156-compression-view/VERIFICATION.json) | `c21413fbfd4c4280aee6103a711105f54359638c` |
| MA-160 | PROMISING | [report](../../experiments/mirror_applications/ma-160-compression-view/README.md) | [verification](../../experiments/mirror_applications/ma-160-compression-view/VERIFICATION.json) | `269ee343d78f7791cb7b0a272834da05ea3cc486` |
| MA-171 | PROMISING | [report](../../experiments/mirror_applications/ma-171-circular-convolution-mirror/README.md) | [verification](../../experiments/mirror_applications/ma-171-circular-convolution-mirror/VERIFICATION.json) | `4025b477f666bb7369531296eb4807c206799b88` |
| MA-173 | PROMISING | [report](../../experiments/mirror_applications/ma-173-fft-phase-mirror/README.md) | [verification](../../experiments/mirror_applications/ma-173-fft-phase-mirror/VERIFICATION.json) | `20adace6cbc1e149265da9d8626897b8d47ddbb4` |
| MA-181 | PROMISING | [report](../../experiments/mirror_applications/ma-181-block-circulant-mirror/README.md) | [verification](../../experiments/mirror_applications/ma-181-block-circulant-mirror/VERIFICATION.json) | `09fadf2f2908ce81742d2de09ea6ffa42b40cc30` |
| MA-186 | FAIL | [report](../../experiments/mirror_applications/ma-186-continual-views/README.md) | [verification](../../experiments/mirror_applications/ma-186-continual-views/VERIFICATION.json) | `e2a54862a926f024254b1e29cc2101c5bdf826b9` |
| MA-189 | PROMISING | [report](../../experiments/mirror_applications/ma-189-freeze-backbone-mirror/README.md) | [verification](../../experiments/mirror_applications/ma-189-freeze-backbone-mirror/VERIFICATION.json) | `ba9639983b09eb75237a6732d1f3c75ad8d75b8a` |
| MA-199 | FAIL | [report](../../experiments/mirror_applications/ma-199-gradient-coordinate/README.md) | [verification](../../experiments/mirror_applications/ma-199-gradient-coordinate/VERIFICATION.json) | `577ceb9f5d8d3d55f8350808e03bb5ee3839e090` |
| MA-208 | FAIL | [report](../../experiments/mirror_applications/ma-208-mirror-expert-distill/README.md) | [verification](../../experiments/mirror_applications/ma-208-mirror-expert-distill/VERIFICATION.json) | `5a37f6abdcb073240541e5fe2349b2ac23454462` |
| MA-241 | PROMISING | [report](../../experiments/mirror_applications/ma-241-expert-tying-mirror/README.md) | [verification](../../experiments/mirror_applications/ma-241-expert-tying-mirror/VERIFICATION.json) | `0ee183668285231d825e853c69c4791b9d252bf2` |
| MA-244 | PROMISING | [report](../../experiments/mirror_applications/ma-244-kv-role-view/README.md) | [verification](../../experiments/mirror_applications/ma-244-kv-role-view/VERIFICATION.json) | `85de2fd65618d72bd0bf6a091b558a0dda57b741` |
| MA-245 | PROMISING | [report](../../experiments/mirror_applications/ma-245-mlkv-layer-views/README.md) | [verification](../../experiments/mirror_applications/ma-245-mlkv-layer-views/VERIFICATION.json) | `76d91b7a662b4227e7f25e4733e06d6735cf1cd2` |
| MA-247 | FAIL | [report](../../experiments/mirror_applications/ma-247-recursive-depth-view/README.md) | [verification](../../experiments/mirror_applications/ma-247-recursive-depth-view/VERIFICATION.json) | `62f0acf3f680ff3bbab8e0e20e194f4065f026e5` |
| MA-248 | FAIL | [report](../../experiments/mirror_applications/ma-248-packet-mirror-code/README.md) | [verification](../../experiments/mirror_applications/ma-248-packet-mirror-code/VERIFICATION.json) | `433a2229362db23bcd43b3830358e674799c6247` |
| MA-249 | PROMISING | [report](../../experiments/mirror_applications/ma-249-future-head-views/README.md) | [verification](../../experiments/mirror_applications/ma-249-future-head-views/VERIFICATION.json) | `20a0984965e879fed8c1b085f02fa5aefc68828b` |
| MA-250 | PROMISING | [report](../../experiments/mirror_applications/ma-250-vsa-expert-address/README.md) | [verification](../../experiments/mirror_applications/ma-250-vsa-expert-address/VERIFICATION.json) | `05ad4f7efb92a3b7bbe8f4f0674377223d6bc768` |
| MA-251 | PROMISING | [report](../../experiments/mirror_applications/ma-251-expert-depth-factorization/README.md) | [verification](../../experiments/mirror_applications/ma-251-expert-depth-factorization/VERIFICATION.json) | `f494b8986a3807e512d0255811f62857758c7ca6` |
| MA-253 | FAIL | [report](../../experiments/mirror_applications/ma-253-cache-safe-final-moe/README.md) | [verification](../../experiments/mirror_applications/ma-253-cache-safe-final-moe/VERIFICATION.json) | `1891cbc36d3a99b4dd63517b469f8b246dbf0be0` |
| MA-691 | PROMISING | [report](../../experiments/mirror_applications/ma-691-lazy-kv-mirror/README.md) | [verification](../../experiments/mirror_applications/ma-691-lazy-kv-mirror/VERIFICATION.json) | `3643118351eb026c31fc00802e47381e8cdfba93` |

## Scientific interpretation and audit cautions

1. The strongest replicated mechanism pattern is **aligned View/orbit variation**: related logical functions can share a physical object plus a small functional coordinate `m`.
2. **Independent/misaligned functions** often require private residual parameters or unrestricted experts. Logical role-count is not independent capacity.
3. Real deployment/runtime remains a barrier: many eager CPU transformations are slower despite lower parameter bytes or theoretical active-MAC proxies.
4. The MA-691 exact canonical-cache primitive is a different evidence class from learned functional compression: it validates algebraic cache reuse under defined transforms, not natural-language throughput.
5. Comparison quality matters: MA-013 was dominated by simpler FiLM, MA-248 by shared-code broadcast, and MA-061/063 by MQA.
6. Protocol deviations remain visible and scoped: MA-129 and MA-208 opened out-of-gate exploratory fresh runs; MA-171 preserved an incorrect-LR exploratory split followed by a preregistered corrected fresh amendment. Do not relabel exploratory results as confirmatory.
7. No test was rerun by this integration operation. The original experiment verification artifacts and recorded replay results were **copied and structurally checked**, not independently re-executed here.

## Next worker action

- Start from `research/mirror-application-worker-ready-20261007`, not from a frozen 254-row experiment snapshot.
- Re-read `STATUS_BOARD.md`, `WORKER_QUEUE.md`, `GOAL.md`, and the Mirror parameter doctrine/matrix.
- Next high-information cross-over P0: **MA-255** (Parameter Superposition with a Mirror context coordinate). Compare genuine Parameter Superposition and ordinary shared/low-rank codes at matched storage and compute.
- Before claiming an ID, check all existing `research/ma-*` branches and the 875-row status.
- Leave prior experiment branches and the default `main` unchanged.
