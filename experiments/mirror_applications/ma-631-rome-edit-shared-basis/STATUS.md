# MA-631 status

- Status: FAIL for Mirror-specific attribution; scoped edit-state compression signal
- Branch: `research/ma-631-rome-edit-shared-basis-20261103`
- Base commit: `ec506ddb`
- Protocol frozen before development: yes
- Development complete: yes (63101, 63102)
- Fresh/audit opened: no; native additive factorization exactly aliases Mirror
- Results committed: pending
- Verification committed: pending

## H / T / D / C / U

**H:** A shared rank-2 basis plus factor-addressed codes stores held-out compositional ROME-style edits with lower actual bytes while preserving efficacy and locality; Mirror must beat ordinary additive coefficient factorization.

**T:** Synthetic 32D linear layer; 16 rank-one edits on 4×4 factor pairs, four diagonal pairs held out. Fit basis and factor tables on 12 training edits. Compared independent rank-one updates, shared low-rank per-edit coefficients, factorized Mirror tables, and native additive factor tables. Measured complete `.npz` payloads, edit-state payloads, efficacy and off-key leakage on two development seeds.

**D — FAIL for Mirror-specific claim.** All methods achieved held-out efficacy nRMSE below 2e-14 and locality leakage 0. Independent full payloads were 6,626–6,648 B; Mirror factor tables were 5,390–5,399 B (about 81%). Edit-state payload was 1,358–1,360 B versus 2,587–2,616 B independent. Shared low-rank per-edit codes were smaller than Mirror (5,277–5,285 B full payload; 1,245–1,246 B edit state). Native additive factor tables exactly matched Mirror arrays, metrics, and bytes. The frozen 50% complete-payload gate failed; fresh remained sealed.

**C — Strongest counter-hypothesis.** The useful representation is ordinary rank-2 update tying plus additive coefficient tables. A generic shared low-rank basis with per-edit coefficients is already smaller than the proposed Mirror table in this setup.

**U — Unconfirmed.** No pretrained Transformer, factuality/generalization tests, real edit requests, or interference under non-orthogonal keys were tested. The one-hot keys make locality exact by construction.

## Fact / Interpretation / Hypothesis

**Fact:** factor-table codes recover held-out rank-one update vectors exactly at lower edit-state bytes than independent vectors, but their full payload reduction is only ~19%; native additive factorization is an exact alias and plain shared low-rank is smaller.

**Interpretation:** shared low-rank update bases can compress this aligned synthetic edit bank, but the Mirror factor coordinate adds no distinct function and misses the frozen full-payload gate.

**Hypothesis:** a real pretrained edit bank with non-orthogonal keys may show whether factor sharing preserves locality better than generic low-rank compression. This remains untested.
