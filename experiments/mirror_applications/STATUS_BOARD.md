# Mirror Application Status Board

Updated: 2026-10-07

## Program totals

- Registered candidates: **254**
- P0: **34**
- P1: **126**
- P2: **94**
- Current MA statuses: **223 UNTESTED, 17 PROMISING, 14 FAIL**
- Historical evidence lanes SRM/TM are not MA statuses.

## Next candidate

**MA-189 — next eligible P0 candidate (Family F)**

Why next:
- MA-241, MA-244, MA-245, MA-247, MA-248, MA-249, MA-250, MA-251, MA-253, MA-003, MA-005, MA-009, MA-019, MA-024, MA-041, MA-048, MA-061, MA-063, MA-076, MA-079, MA-086, MA-116, MA-121, MA-129, MA-156, MA-160, MA-171, MA-173 and MA-181 are checked; Family B KV views remain paused, so MA-186 is next.
- MA-186 completed with an aligned storage/retention benefit but failed the registered quality gate; MA-189 is next in Family F.

If MA-186 is blocked, use the next eligible P0 in the registry.

## Active experiments

None.


When a worker starts an MA experiment, add:
- MA ID;
- branch;
- experiment directory;
- worker/run identifier if available;
- start commit.

- MA-003 — PROMISING: aligned synthetic top-1 Mirror experts passed routed-MSE gate in 2/3 fresh worlds with 35.8% fewer serialized bytes; independent experts needed private capacity; CPU inference was slower. Branch `research/ma-003-mirror-topk-expert-20261007`; report `experiments/mirror_applications/ma-003-mirror-topk-expert/README.md`; result commit `e50a20fe4c000ffb3113f9d3e6564b3efacd4395`.
- MA-005 — PROMISING: signed Givens expert mixture passed aligned quality/storage 3/3 with 41.1% fewer serialized bytes; arbitrary experts required private capacity; current CPU runtime regressed sharply. Branch `research/ma-005-signed-mirror-mixture-20261007`; report `experiments/mirror_applications/ma-005-signed-mirror-mixture/README.md`; result commit `de68ec4c440e54fc4870734e9bbd1fdc99515628`.
- MA-009 — FAIL: one private rare-role expert plus common Mirror views used 0.774x full-MoE bytes and beat shared-only controls, but missed full-MoE relative MSE in 2/3 fresh worlds; arbitrary experts needed more capacity and CPU runtime regressed. Branch `research/ma-009-rare-private-mirror-expert-20261007`; report `experiments/mirror_applications/ma-009-rare-private-mirror-expert/README.md`; result commit `41020ba0aa45d9337fec68f7f772d8d5076534a6`.
- MA-019 — FAIL at development: one-angle Mirror missed the full-MoE quality/storage gates; generic rank-2 basis used only 64B more and fit the aligned teacher much better at the same compute proxy. Fresh worlds were not opened. Branch `research/ma-019-mirror-coefficient-basis-20261007`; report `experiments/mirror_applications/ma-019-mirror-coefficient-basis/README.md`; result commit `b880e34edb8210ad318091be2e17e170ef09c5a0`.
- MA-024 — FAIL at development: one-angle virtual LoRA used 2,401B vs 2,657B generic shared basis, but generic basis had much lower MSE with the same compute proxy; fresh worlds not opened. Branch `research/ma-024-virtual-lora-mirror-20261007`; report `experiments/mirror_applications/ma-024-virtual-lora-mirror/README.md`; result commit `4a41b4af333d181bd873f2f79751177f8c1c9a95`.
- MA-041 — PROMISING: aligned attention output/payload gates passed 3/3 with 32.8% fewer bytes; head-level contribution quality and CPU runtime lagged, independent QKV required private weights. Branch `research/ma-041-mirror-attention-heads-20261007`; report `experiments/mirror_applications/ma-041-mirror-attention-heads/README.md`; result commit `359111bc9cc649fe29ad8793035c237dbf211cba`.
- MA-048 — FAIL: physical-4 to logical-16 Mirror heads saved 43.5% payload and had better head-contribution audit, but aggregate MSE missed full MHA 3/3; CPU throughput 0.189x. Branch `research/ma-048-physical4-logical16-attention-20261007`; report `experiments/mirror_applications/ma-048-physical4-logical16-attention/README.md`; result commit `bdd1dd3002b493cf1355dcc297e9de23b5e0b532`.
- MA-061 — FAIL at development: MQA matched the one-head cache size, had smaller model payload, equal compute proxy, much lower aligned MSE and higher CPU throughput. Fresh worlds stayed sealed. Branch `research/ma-061-single-kv-logical-views-20261007`; report `experiments/mirror_applications/ma-061-single-kv-logical-views/README.md`; result commit `90ae87a3c36d4a9cee79dca01881cd060266ab7b`.
- MA-063 — FAIL at development, confirming MA-061 MQA dominance: equal cache bytes, lower MQA model payload, better MQA output MSE, and no compute advantage for Mirror. Branch `research/ma-063-causal-mirror-mqa-20261007`; report `experiments/mirror_applications/ma-063-causal-mirror-mqa/README.md`; result commit `b69a19315bcead3ea0d05e22d14c76a6e00c63b1`. See `experiments/mirror_applications/FAMILY_B_KV_DIAGNOSTIC_2026-10-07.md`.
- MA-247 through MA-251 are complete and verified.

## Recently completed

- MA-186 — FAIL for the registered quality gate: task-only one-angle views used 20 B/skill vs 241 B/skill rank-2 LoRA and retained aligned skills in 3/3 fresh worlds, but final MSE exceeded the 1.10x LoRA limit in 2/3. Unrelated task maps required private parameters. The shared hypernetwork control was degenerate due zero initialization, so Mirror-specific superiority is not established. Branch `research/ma-186-continual-views-20261007`; report `experiments/mirror_applications/ma-186-continual-views/README.md`; result commit `e2a54862a926f024254b1e29cc2101c5bdf826b9`.

- MA-156 — PROMISING storage/quality frontier: one packed int4 base plus charged views matched independent-int4 quality in 3/3 fresh worlds with 68.0% fewer bytes; stricter hard-tie margin missed, arbitrary matrices needed private QER capacity, decode MAC proxy rose 32x. Branch `research/ma-156-compression-view-20261007`; report `experiments/mirror_applications/ma-156-compression-view/README.md`; result commit `c21413fbfd4c4280aee6103a711105f54359638c`.

- MA-129 — FAIL at development actual-byte gate: Mirror passed aligned quality but payload 1,957B exceeded MTP 1,833B; three exploratory worlds were mistakenly opened after gate failure and are excluded from status. Branch `research/ma-129-temporal-view-20261007`; report `experiments/mirror_applications/ma-129-temporal-view/README.md`; result commit `35cd8bb9fa638006828bdd61ae9511df11dc8af1`.

- MA-121 — PROMISING on the aligned synthetic packet task: improved NLL and exact packet accuracy over MTP in 3/3 fresh worlds with 21.3% fewer bytes; rank-2 PTP had better NLL at higher storage; independent slot functions and eager runtime were poor. Branch `research/ma-121-temporal-view-20261007`; report `experiments/mirror_applications/ma-121-temporal-view/README.md`; result commit `0ccd5470dcac07afba3b9cf77bbfbb7cf8e82bfc`.

- MA-116 — FAIL at development: Mirror-RoPE beat scalar scaling on synthetic held-out positions, but independent frequencies were more accurate and had a smaller actual serialized payload (2,021B vs 2,209B). Fresh stayed sealed. Branch `research/ma-116-mirror-rope-20261007`; report `experiments/mirror_applications/ma-116-mirror-rope/README.md`; result commit `b5d0b623df16f2ea3b024e92c53edb96f26ea00f`.

- MA-086 — FAIL at development storage gate: group-size-2 Mirror matched aligned quality but used 0.777x untied bytes vs required ≤0.65; group-size 4 compressed more but quality fell. Fresh stayed sealed. Branch `research/ma-086-depth-address-20261007`; report `experiments/mirror_applications/ma-086-depth-address/README.md`; result commit `155231548d0c202ec1d23b708206452434a7c2b7`.

- MA-079 — PROMISING for sparse depth interpolation on the aligned teacher: 3/3 fresh worlds recovered held-out maps at median MSE 3.36e-12 with 2,149B vs 3,753B sparse-trained untied and 2,213B generated gain. Independent maps did not benefit; untied held-out quality is not a fully supervised upper control. Branch `research/ma-079-learned-depth-address-20261007`; report `experiments/mirror_applications/ma-079-learned-depth-address/README.md`; result commit `f1cf8d6a64e7f3fbee4b25bddfba71f0a8f731db`.

- MA-076 — PROMISING on the deliberately Givens-aligned depth teacher: matched untied quality in 3/3 fresh worlds with 21.3% fewer payload bytes; independent layer maps remained poorly fit and eager CPU throughput was 0.235x untied. Branch `research/ma-076-one-block-many-layers-20261007`; report `experiments/mirror_applications/ma-076-one-block-many-layers/README.md`; result commit `418a888bfebfb9bf0ad38dbc1b3c97a5001ec7ce`.

- MA-251 — PROMISING: factorized Givens expert/depth coordinates matched untied quality 3/3 with 76.4% fewer bytes; independent pair functions required more capacity; CPU throughput regression recorded. Branch `research/ma-251-expert-depth-factorization-20261007`; report `experiments/mirror_applications/ma-251-expert-depth-factorization/README.md`; result commit `f494b8986a3807e512d0255811f62857758c7ca6`.

- MA-250 — PROMISING on aligned linear expert roles: Mirror matched untied quality 3/3 with 38.3% fewer payload bytes; independent roles required private/full weights; fixed MAP/Hadamard/HRR codes did not fit this Givens-aligned teacher. Branch `research/ma-250-vsa-expert-address-20261007`; report `experiments/mirror_applications/ma-250-vsa-expert-address/README.md`; result commit `05ad4f7efb92a3b7bbe8f4f0674377223d6bc768`.

- MA-249 — PROMISING on aligned synthetic head sharing: 3/3 fresh worlds matched MTP top-1 with 32.5% lower full model payload; CPU inference throughput 0.42x MTP; independent-head control required more private degrees of freedom. Branch `research/ma-249-future-head-views-20261007`; report `experiments/mirror_applications/ma-249-future-head-views/README.md`; result commit `20a0984965e879fed8c1b085f02fa5aefc68828b`.

- MA-248 — FAIL for Mirror-specific frontier: packet Givens views passed 2/3 correlated fresh worlds, while broadcast shared code passed 3/3 at 378 fewer serialized bytes; independent entropy boundary NOT ESTABLISHED. Branch `research/ma-248-packet-mirror-code-20261007`; report `experiments/mirror_applications/ma-248-packet-mirror-code/README.md`; result commit `433a2229362db23bcd43b3830358e674799c6247`.

- MA-247 — FAIL at development screen; Mirror used fewer bytes than untied but had worse MSE than tied, scalar-gate, and static LoRA controls. Fresh worlds were not opened by the failure rule; branch `research/ma-247-recursive-depth-view-20261007`; report `experiments/mirror_applications/ma-247-recursive-depth-view/README.md`; result commit `62f0acf3f680ff3bbab8e0e20e194f4065f026e5`.

- MA-245 — PROMISING (aligned output/cache result; missed model-payload threshold; CPU slowdown); branch `research/ma-245-mlkv-layer-views-20261007`; report `experiments/mirror_applications/ma-245-mlkv-layer-views/README.md`; result commit `76d91b7a662b4227e7f25e4733e06d6735cf1cd2`.
- MA-244 — PROMISING (aligned quality/cache mechanism; missed 20% model-payload gate; CPU slowdown); branch `research/ma-244-kv-role-view-20261007`; report `experiments/mirror_applications/ma-244-kv-role-view/README.md`; result commit `85de2fd65618d72bd0bf6a091b558a0dda57b741`.
- MA-253 — FAIL for Mirror expert replacement; cache-placement mechanics PASS; branch `research/ma-253-cache-safe-final-moe-20261007`; report `experiments/mirror_applications/ma-253-cache-safe-final-moe/README.md`; result commit `1891cbc36d3a99b4dd63517b469f8b246dbf0be0`.
- MA-241 — PROMISING (3/3 synthetic quality/storage gate; CPU runtime regression); branch `research/ma-241-expert-tying-mirror-20261007`; report `experiments/mirror_applications/ma-241-expert-tying-mirror/README.md`; result commit `0ee183668285231d825e853c69c4791b9d252bf2`.

SRM001–003 and TM001 are predecessor evidence and remain in their own namespaces.

## Blocked

None.

## Status policy

The authoritative scientific status is the registry row. This board is an operational cache. If they disagree, fix the board from the registry, not the other way around.

- MA-160 — PROMISING, with strict fresh quality gate narrowly missed: shared residual Mirror used 393B vs 607B independent int4 (0.647x) and averaged 1.035x fresh activation MSE, but seed 16011 was 1.106x against a 1.10x limit. Residual-free Mirror averaged 3.37x MSE; independent role matrices needed private state. Branch `research/ma-160-compression-view-20261007`; report `experiments/mirror_applications/ma-160-compression-view/README.md`; result commit `269ee343d78f7791cb7b0a272834da05ea3cc486`.

- MA-171 — PROMISING on an intentionally HRR-aligned linear teacher: corrected fresh seeds 17121–17123 passed the preregistered quality/byte gate with HRR MSE below fixed-update untied and 2,853B vs 5,349B (0.533x). Arbitrary role functions required private weights; not capacity evidence. Original fresh seeds were mistakenly run at LR 0.01 instead of dev-selected 0.003, retained as exploratory and excluded; corrected seed amendment A1 was frozen before access. Branch `research/ma-171-compression-view-20261007`; report `experiments/mirror_applications/ma-171-circular-convolution-mirror/README.md`; result commit `4025b477f666bb7369531296eb4807c206799b88`.

- MA-173 — PROMISING on a learned real Fourier-phase teacher: 3,169B vs 5,349B untied (0.592x) and lower fixed-update MSE in 3/3 fresh aligned worlds. Fixed HRR/MAP/Hadamard/Givens did not fit this orbit; independent roles required private parameters. Active MAC proxy improved, but measured training wall and inference throughput were slower. Branch `research/ma-173-holographic-compression-20261007`; report `experiments/mirror_applications/ma-173-fft-phase-mirror/README.md`; result commit `20adace6cbc1e149265da9d8626897b8d47ddbb4`.

- MA-181 — PROMISING aligned storage/quality frontier: shared block-circulant kernel bank plus role shifts passed quality/bytes in 3/3 fresh worlds at 2,657B vs 6,437B untied (0.413x), and 21% fewer bytes than independent block-circulant at similar MSE. Unrelated roles required dense private parameters. Active MAC proxy slightly exceeded untied; eager training/inference runtime regressed sharply. Branch `research/ma-181-compression-view-20261007`; report `experiments/mirror_applications/ma-181-block-circulant-mirror/README.md`; result commit `09fadf2f2908ce81742d2de09ea6ffa42b40cc30`.
