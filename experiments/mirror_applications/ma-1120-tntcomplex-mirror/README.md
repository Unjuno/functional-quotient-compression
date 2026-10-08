# MA-1120 — TNTComplEx relation x time Mirror factor bank

## H — hypothesis

A shared temporal relation basis with compact per-relation Mirror coordinates could preserve future relation-time links on a low-dimensional periodic family with substantially fewer inference bytes than native temporal factors, while independently sampled operators would require private state.

## T — execution

CPU synthetic temporal graph, 24 entities, 6 relations, rank 8; training timestamps 0–7, development 8–9, fresh audit 10–11. Fresh worlds: 112101, 112102, 112103. Each method received 120 Adam updates and 11,520 sampled training examples. The same fixed sinusoidal 8-period time features were used by every method. Reported metrics use filtered entity ranking over both known positive tails for each `(head, relation, time)`.

Controls: native periodic relation modulation; a TuckER-style relation coefficient × time feature × shared basis; per-relation low-rank time maps; and independent relation-time operators as a diagnostic. The last is not a valid future-time upper control because its timestamp-specific parameters for audit times are untrained.

Actual inference state was serialized as safetensors. Parameters unused by the scoring path were removed before the final audit. Earlier metric and payload runs are retained under `source/` as provenance; final interpretation uses `source/audit_results.json`.

## D — FAIL

On aligned fresh worlds, Mirror and TuckER produced identical filtered MRR and Hits@1 in all 3 seeds. Mean filtered MRR was 0.21743 for both, versus 0.19239 for native; this is a shared low-rank/Tucker benefit, not a Mirror-specific result. The Mirror payload was 1,264 B versus 1,280 B native (1.25% smaller), far above the preregistered `<=60%` storage gate. The Mirror and TuckER payloads were identical in size. Compute was similar: mean training wall time 0.055 s Mirror, 0.061 s TuckER, 0.058 s native; eager CPU all-tail inference was about 34.9, 35.0, and 48.5 microseconds per example respectively. Tiny synthetic timings are not deployment claims.

## C — strongest counter-hypothesis

The Mirror parameterization is algebraically equivalent to an ordinary TuckER-style low-rank relation-time factorization. The exact equality across all aligned audit seeds supports this counter-hypothesis. The observed quality gain over native comes from the shared low-rank basis, not Mirror coordinates.

## U — unconfirmed

No benchmark or natural temporal knowledge graph was tested. The low filtered MRR values indicate weak absolute task performance. Independent operators could not serve as a valid unseen-time upper bound under the chronological split. No capacity claim is supported; all methods used a fixed update budget.

## Fact / Interpretation / Hypothesis

- Fact: Mirror and TuckER matched exactly in filtered MRR, Hits@1, and bytes on each aligned fresh world; Mirror saved 16 B versus native.
- Interpretation: the chosen Mirror insertion duplicates a standard factorized temporal tensor model and does not establish Mirror-specific gain or the storage target.
- Hypothesis: stronger results may require a different temporal address construction and a valid native extrapolator control; this experiment gives no evidence for that direction.
