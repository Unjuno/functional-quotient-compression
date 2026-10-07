# Mirror Application Status Board

Updated: 2026-10-07

## Program totals

- Registered candidates: **254**
- P0: **34**
- P1: **126**
- P2: **94**
- Current MA statuses: **208 UNTESTED, 28 PROMISING, 18 FAIL**
- Historical evidence lanes SRM/TM are not MA statuses.

## Next candidate

**MA-016 — next eligible candidate (MoE / experts)**

Why next:
- MA-010 and MA-011 failed at pre-fresh development gates. MA-012 passed its aligned residual-view gate 3/3 but CPU throughput regressed and independent roles needed private capacity. MA-013 passed aligned quality/storage vs independent, but FiLM dominated the Mirror frontier. MA-014 extended MA-241 to four layers and top-1 sparse execution; aligned quality/storage passed 3/3, with CPU slowdown. MA-015 passed domain×expert factorization in 3/3 aligned fresh worlds, with CPU slowdown and an independent-function boundary. MA-016 is next UNTESTED in registry order.

## Active experiments

None. MA-008 and MA-010 through MA-015 are complete; see result entries below.

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
- MA-015 — PROMISING: three-domain × four-expert factorized Mirror views passed aligned quality/storage 3/3 at 18,727B vs 51,060B independent (0.367x); MSE 0.503–0.621x full. It used 2.8% more bytes than hard tying and reduced MSE 37.5–60.6%; rank-2 residual used more bytes for slightly higher MSE. CPU throughput was 0.55x tied. Independent functions needed private state. Branch `research/ma-015-domain-factorized-experts-20261007`; report `experiments/mirror_applications/ma-015-domain-factorized-experts/README.md`; result commit `65e006d1f4b2fac64c4b89d89437e36f481f6ea9`.

- MA-014 — PROMISING: four-layer oracle top-1 Mirror views passed aligned quality/storage 3/3 at 6,372B vs 18,289B independent (0.348x); MSE was 0.384–0.508x full. It used 7.5% more bytes than hard tying and had 86.7–92.6% lower MSE, but CPU throughput was 0.63x tied. Extension of MA-241; natural MoE untested. Branch `research/ma-014-layer-specific-expert-bank-20261007`; report `experiments/mirror_applications/ma-014-layer-specific-expert-bank/README.md`; result commit `bf54b738a1e88abd0165f1fc889b1639b262d05b`.

- MA-013 — FAIL for Mirror-specific frontier: aligned quality/storage vs independent passed 3/3 (0.384x payload), but multiplicative Mirror used 18.3% more bytes than hard tying; FiLM used 382B fewer and had lower MSE in all three fresh worlds. Independent functions needed private state. Branch `research/ma-013-multiplicative-mirror-moe-20261007`; report `experiments/mirror_applications/ma-013-multiplicative-mirror-moe/README.md`; result commit `e0c7831f0547de1bb824cbe403f01bb9138e0b4d`.

- MA-012 — PROMISING: shared Mirror plus rank-2 private residual passed aligned quality/storage and matched-rank comparison in 3/3 fresh worlds (0.458x full-expert bytes; 0.393–0.597x MSE). It used 6.5% more bytes than tied-rank2 for 18.7–33.1% lower MSE; CPU throughput was 0.52x tied-rank2. Independent roles required richer state. Branch `research/ma-012-residual-mirror-moe-20261007`; report `experiments/mirror_applications/ma-012-residual-mirror-moe/README.md`; result commit `3cdff5dc56cc2c44a1b29112d823c9e13ee9744a`.

- MA-011 — FAIL at development storage gate: aligned product Mirror achieved 0.140x independent MSE and lower error than rank-2 at fewer bytes, but payload was 0.719x vs required <=0.65x; fresh stayed sealed. Independent factor functions needed richer state; CPU throughput lagged. Branch `research/ma-011-product-mirror-experts-20261007`; report `experiments/mirror_applications/ma-011-product-mirror-experts/README.md`; result commit `5ae711ee4e6721ef1cee6e78e666b7f760f609d9`.

- MA-010 — FAIL at development: sequential Mirror expert composition missed both pre-fresh gates (MSE 2.13x independent; payload 0.875x vs required 0.65x). Rank-2 residual had lower aligned MSE; hard tying was smaller. Fresh stayed sealed. Branch `research/ma-010-sequential-mirror-experts-20261007`; report `experiments/mirror_applications/ma-010-sequential-mirror-experts/README.md`; result commit `64caf2d8e5f338a53347f6b59628fbc6ad30fced`.


- MA-007 — PROMISING: balanced-role token-choice Mirror passed aligned quality/storage 3/3 with full coverage (Mirror/untied MSE 0.766–0.769; payload 0.334x). Paired expert-choice Mirror coverage was 0.818–0.827 and MSE 3.3–4.8x worse. Hard tying was 252B smaller; Mirror throughput was 0.233x tied. Branch `research/ma-007-token-choice-mirror-20261007`; report `experiments/mirror_applications/ma-007-token-choice-mirror/README.md`; result commit `877e22589af90afb3ddd0e7f422977034831bab7`.

- MA-006 — PROMISING for the registered expert-choice sharing gate: Mirror passed 3/3 aligned quality/coverage/storage at 7,761B vs 23,341B full expert-choice. Token-choice had 100% coverage and 2.50–3.42x lower MSE; no-route expert-choice tokens and eager runtime limit usefulness. Mirror-specific FiLM/residual margin failed. Branch `research/ma-006-expert-choice-mirror-20261007`; report `experiments/mirror_applications/ma-006-expert-choice-mirror/README.md`; result commit `572e245bee17ffe430863a4827baa21da809fb82`.

- MA-004 — PROMISING: nonlinear dense softmax Givens mixture passed aligned quality/storage 3/3 (Mirror/untied MSE 0.297–1.030; 7,697B vs 23,277B). Hard tying was 252B smaller and used one quarter of active MACs; independent roles needed private/richer state, and Mirror CPU throughput was 0.086x tying. Synthetic fixed-update result. Branch `research/ma-004-soft-mirror-expert-mixture-20261007`; report `experiments/mirror_applications/ma-004-soft-mirror-expert-mixture/README.md`; result commit `e8fea8ef10f876835c4683bad0203ec7fdc91e16`.

- MA-002 — PROMISING: nonlinear sparse top-2 Givens views passed aligned quality/storage 3/3 (Mirror/untied MSE 0.223–0.403; 7,697B vs 23,277B); hard tying was 252B smaller and ~2.03x lower active proxy. Independent functions needed private/richer state; Mirror CPU inference throughput was 0.102x tying. Synthetic fixed-update result; protocol base_commit typo is disclosed. Branch `research/ma-002-mirror-top2-expert-20261007`; report `experiments/mirror_applications/ma-002-mirror-top2-expert/README.md`; result commit `2b7f7fe736393ef3702daa7995e0951d54e4de12`.

- MA-001 — PROMISING: nonlinear single-layer top-1 Givens views passed the aligned quality/storage gate in 3/3 fresh worlds (Mirror/untied MSE 0.419–0.572; 7,697B vs 23,277B). Hard tying used 252B fewer bytes, Mirror inference throughput was 0.194x hard tying, and unrelated functions needed private capacity. Synthetic fixed-update result only. Branch `research/ma-001-mirror-top1-expert-20261007`; report `experiments/mirror_applications/ma-001-nonlinear-top1-expert/README.md`; result commit `60ca242afcce0ff78b5116ce57dbe454a9dc4629`.

- MA-111 — PROMISING: the registered aligned held-out-filler quality/storage gate passed in 3/3 fresh worlds at 2,770B vs 5,839B full per-role maps (0.474x); rank-2 output LoRA matched quality at 3,411B. Mirror active-compute proxy was 0.773x LoRA but measured throughput was 0.599x and train wall 1.25x. Independent random role transforms needed richer/private state; the exact teacher used a packed-angle payload 17B smaller than current Mirror records. Synthetic controlled roles, no natural semantic/LM claim. Branch `research/ma-111-semantic-role-embedding-20261007`; report `experiments/mirror_applications/ma-111-semantic-role-embedding/README.md`; result commit `865c8f1f842db934b94564ed0c1d97f688879740`.

- MA-208 — FAIL at development: Mirror passed absolute KL, top-1, ECE and 0.476x multi-head bytes, but exceeded the relative-KL limit in both aligned seeds (38.8x and 1,027.7x). Fresh seeds 20811–20813 were mistakenly opened after this subgate was overlooked; they are exploratory only and split integrity is false. Exploratory aligned payload was 1,322B vs 2,775B, but throughput was 0.435x multi-head; independent teachers needed private weights. Branch `research/ma-208-mirror-expert-distill-20261007`; report `experiments/mirror_applications/ma-208-mirror-expert-distill/README.md`; result commit `5a37f6abdcb073240541e5fe2349b2ac23454462`.

- MA-199 — FAIL at development: task Givens coordinates used 1,103B total vs 1,196B rank-1 LoRA (0.922x), missing the <=0.90x byte gate in both development seeds; fresh stayed sealed. Incremental bytes/skill were lower, but the generic shared-plane control had similar mean quality and smaller resume state. Independent updates needed private weights. Branch `research/ma-199-gradient-coordinate-20261007`; report `experiments/mirror_applications/ma-199-gradient-coordinate/README.md`; result commit `577ceb9f5d8d3d55f8350808e03bb5ee3839e090`.

- MA-189 — PROMISING on the deliberately two-sided-Givens-aligned task family: at 64 examples, 2-view Mirror passed quality/byte/retention in 3/3 fresh worlds with 55B incremental inference vs 228B rank-2 LoRA and unchanged Task-0 MSE. Unrelated maps needed private state; eager inference throughput was 0.20x LoRA. Rank-4 LoRA and generic byte-matched basis remain untested. Branch `research/ma-189-freeze-backbone-mirror-20261007`; report `experiments/mirror_applications/ma-189-freeze-backbone-mirror/README.md`; result commit `ba9639983b09eb75237a6732d1f3c75ad8d75b8a`.

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
## Recent completed result

- MA-008 — PROMISING for aligned hierarchical expert sharing: passed quality/storage in 3/3 fresh worlds (Mirror/full-hier MSE 0.931–0.949; 8,898B vs 24,673B), but hard tying was 317B smaller and Mirror CPU throughput was 0.317x tied. Hierarchy did not consistently beat flat routing; independent roles needed richer state. Branch `research/ma-008-hierarchical-mirror-moe-20261007`; report `experiments/mirror_applications/ma-008-hierarchical-mirror-moe/README.md`; result commit `2497982dc0d2861cc89acc115cffb2877439c71e`.
