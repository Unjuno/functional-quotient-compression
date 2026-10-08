# MA-674 — Sparse top-k Mirror LoRA atoms

## H — hypothesis

A shared low-rank atom bank plus sparse task codes may lower actual storage and active update compute versus independent functional deltas. It must beat a matched MoLE-style direct mixture for Mirror-specific value.

## T — execution

Synthetic 8×8 functional update matrices. Development task identities exposed six single-atom updates and six mixtures; the shared six-atom bank was fit from development tasks only. Each fresh world had 12 in-span two-atom compositions and 4 private rank-1 out-of-span deltas. Fresh seeds 67411–67413. Codes were oracle-fit from full heldout matrices, not learned from examples; zero optimizer updates. The primary comparison used full delta matrices, avoiding LoRA factor gauge ambiguity. Actual NPZ payload bytes were measured.

## D — FAIL (Mirror-specific gate)

In-span top-2 reconstruction mean matrix MSE was `8.77e-18`. Shared atom + sparse code payload was 2,484 B (56.8% of the 4,372 B independent full-delta bank), and activated two atoms/task. The matched MoLE top-2 method had identical function, bytes and active atom count. It therefore failed the Mirror-specific gate. The dense six-coefficient atom representation used 2,452 B, slightly fewer actual bytes than the sparse index format, though it activates six atoms. Out-of-span top-2 MSE averaged 0.00848. Adding private residuals restored numerical exactness but cost 6,846 B, more than the independent 4,372 B bank.

## C — strongest counter-hypothesis

Sparse atom mixing is ordinary MoLE/shared-basis composition, and small-array serialization overhead can erase index-level savings. The observed in-span storage result reflects a deliberately aligned synthetic basis, not learned task transfer.

## U — unconfirmed

No pretrained adapters, task-example adaptation, optimizer training, downstream quality or inference throughput were tested. Out-of-span private residual efficiency remains poor in this representation.

## Fact / Interpretation / Hypothesis

- **Fact:** top-2 reconstruction is near exact in span; MoLE top-2 is identical; out-of-span residuals raise bytes above independent storage.
- **Interpretation:** sparse shared atoms compress aligned functions, but do not add Mirror-specific capability and require costly private state off span.
- **Hypothesis:** larger realistic adapter banks with entropy-coded sparse addresses may improve actual bytes; new experiment needed.
