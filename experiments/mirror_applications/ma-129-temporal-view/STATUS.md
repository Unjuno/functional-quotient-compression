# MA-129 status

- Status: FAIL at development gate
- Branch: `research/ma-129-temporal-view-20261007`
- Base commit: `1ccd2583fa05619acbb87ce8f2c2531dce04cfcc`
- Last verified commit: `35cd8bb9fa638006828bdd61ae9511df11dc8af1`
- Development complete: yes; seeds 12901–12902
- Fresh/audit opened: yes, incorrectly after development byte-gate failure; exploratory only
- Results committed: yes (`35cd8bb9fa638006828bdd61ae9511df11dc8af1`)
- Verification committed: yes (`35cd8bb9fa638006828bdd61ae9511df11dc8af1`)
- Registry row updated: yes

## H / T / D / C / U

- **H:** One shared branch verifier plus small Mirror phase codes recovers four correlated branch decisions with lower payload than independent MTP heads and similar valid-path quality.
- **T:** Four synthetic binary decisions per feature vector; tied, rank-2 shared branch code, Mirror, independent MTP, and sequential shared-score controls; two development seeds, then three inadvertently opened exploratory seeds; aligned and independent teacher families; 500 Adam updates at LR 0.02.
- **D:** FAIL: development quality passed, but Mirror payload was 1,957B vs 1,833B MTP, missing the ≤0.80× byte gate in both dev worlds. The fresh-open action was a protocol deviation; those rows are excluded from the decision.
- **C:** `torch.save` tensor metadata erased the learned-scalar count advantage; the aligned teacher also follows the Mirror Givens orbit by construction.
- **U:** A packed inference serializer, genuine KV-cached Transformer verifier, language-level acceptance, and optimized phase kernels.

**FACT:** 50 total rows replayed; exact payload bytes; maximum metric delta 4.95e-11; tests 2/2 pass. First 20 rows are preregistered development; remaining 30 were exploratory after gate failure. Fresh integrity is false.

**INTERPRETATION:** Under this measured serializer, Mirror fails the storage gate despite passing aligned branch quality. Exploratory later quality cannot repair that preregistered decision.

**HYPOTHESIS:** Packed tensor serialization may expose a storage advantage, but must be tested in a separately preregistered candidate.

## Next action

Commit the FAIL and protocol-deviation record, then proceed to MA-156.

## Blockers

None for the synthetic screen; the procedural deviation is disclosed.

## Decisions / rulings

Seeds 12911–12913 were run contrary to the failure gate after development. Retain all rows as exploratory, mark fresh-split integrity false, and exclude them from the scientific status decision.
