# MA-632 status

- Status: FAIL for Mirror-specific attribution and frozen full-payload gate; useful aligned/private frontier map
- Branch: `research/ma-632-rome-private-frontier-20261103`
- Base commit: `ade36fdb`
- Protocol frozen before development: yes
- Development complete: yes (63201, 63202)
- Fresh/audit opened: no; ordinary joint low-rank control dominates
- Results committed: pending
- Verification committed: pending

## H / T / D / C / U

**H:** The rank of private variation determines where shared edit bases stop reducing bytes; a shared/private Mirror split must beat an ordinary joint low-rank basis.

**T:** Synthetic 32D linear layer, 16 rank-one edits, four held out; shared rank 2 and true private ranks 0/2/4/8/16. Swept stored private ranks 0/1/2/4/8/16. Bases fit on 12 edits. Compared independent ROME updates, shared/private Mirror basis and ordinary joint low-rank basis. Two development seeds; efficacy, locality and actual payload bytes measured.

**D — FAIL.** For true private ranks 0, 2, and 4, exact efficacy was recovered with stored ranks 0, 2, and 4 respectively. Mirror full payloads were 5,687–5,693 B, 6,091–6,093 B, and 6,491–6,492 B, versus independent updates at 6,621–6,639 B. The frozen ≤70% gate failed at all aligned ranks. At private rank 8, exact recovery required 7,230–7,238 B, larger than independent updates (~6,624 B). At true rank 16, 12 training edits could not identify the full private subspace; q=16 still had nRMSE 0.276–0.537, while independent rank-one updates were exact. Locality leakage was zero by one-hot key construction. The native joint low-rank control matched quality/locality and used 400–460 fewer complete bytes than the Mirror split throughout. Fresh stayed sealed.

**C — Strongest counter-hypothesis.** The Mirror split is an ordinary shared/private low-rank factorization. A single concatenated basis is more compact than serializing separate shared and private matrices.

**U — Unconfirmed.** No pretrained Transformer, factuality, key interference, or real ROME edit behavior was tested. Higher training-edit coverage might identify more of the private rank-16 span, but would increase training data and cannot improve the tested native-control comparison without a new protocol.

## Fact / Interpretation / Hypothesis

**Fact:** private rank 0/2/4 was recovered exactly at payloads modestly below independent updates; rank 8 eliminated the byte saving; rank 16 could not be recovered from 12 training edits. Native joint low rank had equal quality and lower bytes.

**Interpretation:** private parameters are necessary as edit variation departs from the shared rank-2 space, and their basis/code cost quickly removes the storage benefit. Mirror adds no function beyond standard low-rank factorization.

**Hypothesis:** non-orthogonal keys and real model locality constraints may change the break-even, but need a redesigned family experiment.
