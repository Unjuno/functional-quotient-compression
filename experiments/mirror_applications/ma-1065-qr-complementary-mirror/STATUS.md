# MA-1065 status

Decision: **FAIL (development-only screen; no fresh audit opened)**.

## H — Hypothesis

One shared complementary QR factor pair plus two 16-angle role-specific Givens views can recover useful user/item specialization, exceed the best shared-QR non-Mirror control by 0.003 AUC, and approach independent QR while using fewer complete serialized bytes.

## T — Test

MovieLens 1M rating>=4 binary GMF screen. The deterministic train/dev partitions exclude the MA-1063 and MA-1064 scored pairs. Dev seeds 106500 and 106501 were used for the preregistered 3-epoch screen. A development-only diagnostic then trained independent QR, shared QR, shared QR+Givens, shared QR+FiLM, and full tables through 15 epochs on the same train/dev partition. No fresh audit rows or metrics were loaded. QR tables use complementary quotient/remainder mappings; factor rows and head are learned jointly with Adam.

## D — Decision and facts

At epoch 3, QR/Givens AUC was 0.5000 and logloss 0.68243 on both dev seeds. The 15-epoch diagnostic did not recover them: best-dev-logloss checkpoints had AUC 0.49984/0.50000 for independent QR, 0.50000/0.50000 for shared QR, and 0.50000/0.50000 for shared QR+Givens. Shared QR+FiLM reached AUC 0.76293/0.76486 at epoch 15; full tables reached 0.80837/0.80747. The 15-epoch runs took about 16–17 seconds per FiLM/full-table seed, 16–17 seconds per QR seed, and 35–36 seconds per Givens seed. This is a scoped **FAIL** for the frozen implementation/task screen: the Mirror hypothesis is not supported, and the selected model could not satisfy the quality gates, so no fresh audit was opened.

The original preregistered success gate also refers to independent QR quality and a best shared-QR non-Mirror comparison, while the experiment did not preregister a minimum absolute quality floor. Consequently the relative AUC gate is mathematically vacuous when all QR variants are at chance; the failure decision here relies on the required utility/quality premise, not on treating a vacuous relative margin as evidence. Payload, fresh quality, and fresh throughput comparisons remain unmeasured.

## C — Strongest counter-hypothesis

MovieLens GMF with one positive/negative rating label may be poorly matched to the chosen low-rank QR factorization and shared user/item namespace. The observed collapse may be a task/model/optimization mismatch, not evidence against QR on CTR, optimized QR-Rec, or Mirror views generally. FiLM's role-specific offsets add a direct discriminative signal that the origin-preserving Givens coordinate does not provide.

## U — Unconfirmed

No sealed audit result, serialized inference payload sizes, actual-byte frontier, fresh throughput, exact replay, or broad QR/CTR evidence is established. This is one deterministic MovieLens split and two training seeds. The development diagnostic is exploratory and does not change the frozen audit protocol or claim a capacity result.

Prior audit exclusion counts and split provenance are in `runs/development.json`; the 15-epoch development-only diagnostic is in `runs/development_extended.json`. The failed pre-metric setup attempt is retained at `runs/FAILED_ATTEMPT_DEV01.json`.
