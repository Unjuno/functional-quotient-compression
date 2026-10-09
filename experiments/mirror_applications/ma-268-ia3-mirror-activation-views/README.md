# MA-268 — IA3 versus Mirror activation views

## H — Hypothesis

A one-angle Givens view at the hidden activation interface can recover rotated task functions using fewer task-state bytes than IA3's 32 channel scales. Independent diagonal task variation should favor native IA3. This is synthetic function reconstruction, not an NLP quality claim.

## T — Training and evaluation

Frozen shared feature network (16→32→8), 8 task identities, 128 support and 512 audit examples per task. Development worlds 26800–26801 and fresh worlds 26810–26812, each with seeds 0–2. Compared no view, native IA3, one-angle Givens Mirror in the aligned stratum, and an independent dense activation map upper control. One shared model payload and every task code are charged in a canonical flat float32 serializer. The frozen source/protocol was pushed before fresh execution. Fresh results contain 504 rows. CPU only; no GPU.

## D — Result: FAIL against the registered storage gate; scoped functional result

Fresh means over task/world/seed instances:

| Stratum | Method | Mean NRMSE | Max NRMSE | Payload bytes | Fit seconds | Apply μs |
|---|---|---:|---:|---:|---:|---:|
| aligned_rotation | no_view | 1 | 1 | 3,158 | 0.000000 | 28.0 |
| aligned_rotation | mirror_givens | 8.1579338e-09 | 5.2283681e-08 | 3,167 | 0.029281 | 94.8 |
| aligned_rotation | ia3 | 0.04784988 | 0.11532943 | 3,283 | 0.014094 | 62.0 |
| aligned_rotation | dense_upper | 5.0893073e-07 | 7.399733e-07 | 7,262 | 0.000350 | 42.0 |
| independent_diagonal | no_view | 1 | 1 | 3,158 | 0.000000 | 30.7 |
| independent_diagonal | ia3 | 1.1024168e-06 | 2.0910593e-06 | 3,283 | 0.020128 | 44.2 |
| independent_diagonal | dense_upper | 5.2070438e-07 | 7.1135162e-07 | 7,262 | 0.000346 | 48.3 |

Fact: aligned Mirror payload was 3,167 B vs IA3 3,283 B (3.5% less), missing the preregistered requirement of at most 80% of IA3 bytes. Mirror NRMSE stayed below 5.3e-8 on every fresh aligned task; IA3 mean NRMSE was 0.04785. On independent diagonal tasks, IA3 mean NRMSE was 1.10e-6. Dense upper had near-zero error at 7,262 B.

Interpretation: the view family matters when task transforms lie exactly in its one-angle orbit; the same compact coordinate does not replace independent channel scales. This is not a capacity gain claim. The shared payload dominates bytes, so removing 31 float coordinates saves only 116 B at this model size.

Hypothesis: at larger hidden width or with shared weights amortized over more tasks, angle codes may yield more meaningful state reduction; a native one-parameter rotation control or packed code may erase the small advantage.

## C — Strongest counter-hypothesis

The result is a parameterization match: the aligned teacher used the same one-angle Givens family. IA3 is structurally mismatched to a rotation, so the quality gap establishes orbit recovery, not general Mirror superiority. Payload reduction is marginal and Mirror fitting took about twice as long as IA3 in this CPU harness.

## U — Not established

No transformer/NLP task, end-to-end training, larger widths, multiple rotation planes, learned angle quantization, multi-task amortization frontier, or GPU runtime was measured. No Mirror-specific novelty claim is made.
