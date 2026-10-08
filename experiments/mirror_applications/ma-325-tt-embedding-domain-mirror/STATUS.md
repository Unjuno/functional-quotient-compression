# MA-325 status

- Status: FAIL under A3 development gate; original A0 task was NOT ESTABLISHED; fresh remains sealed.
- Branch: `research/ma-325-tt-embedding-domain-mirror-20261008`
- Base commit: `1edb52e`
- Fresh seeds 32511–32513 sealed.
- Development seeds 32501/32502 executed; across six methods test NLL remained approximately ln(16) and accuracy approximately 1/16, including independent full embeddings. Thus the task is not being learned and cannot decide the Mirror hypothesis.
- Source bug found and fixed before any fresh access: Mirror phase vector now includes the deliberately unrelated third domain. Two exploratory teacher rescalings remained unlearnable and are not confirmatory evidence; the frozen task/gates were not revised for fresh evaluation.
- No verdict is assigned to Mirror. A3 changes only teacher logit scale to 32 based on development-validation learnability diagnostics; confirmatory development must pass independent-full gate before fresh is opened.

A3 amendment selected on development validation only: original scale 8 produced independent-full macro NLL ~2.74 and accuracy ~0.096; scale 32 made the task learnable. Confirmatory development passes quality but fails storage (2631B vs 2637B direct, required <=2241B). Fresh IDs 32511-32513 remain sealed. The original evidence is retained separately.

## A3 disposition

- **H:** domain Mirror phase should preserve learnable aligned domains while using <=85% of direct-coefficient bytes.
- **T:** two development seeds, six methods, 600 updates; A3 changes teacher logit scale only. 12 payloads reloaded, hashes/validation/test metrics replay exactly, four tests pass.
- **D:** FAIL for Mirror-specific byte gate; no fresh opening.
- **C:** ordinary direct coefficients have effectively identical quality and bytes.
- **U:** fresh replication and natural-language/Transformer evidence remain untested.
