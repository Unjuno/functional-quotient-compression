# MA-817 — add a new what/content operand

## H / T / D / C / U

**H:** A new what can be added using only a small code on frozen shared content factors while preserving existing content and beating explicit storage/retraining.

**T:** Synthetic 8D diagonal mechanisms, six operands, three operations; rank-2 frozen content basis; one new operand inferred from 32 support samples and scored over 256 examples per operation; seed 81701; zero optimizer updates.

**D: FAIL.** Mirror code gives heldout nRMSE 9.99e-8 and retains old operands, but the complete payload is 1,241 B versus 1,125 B for explicit new-what storage and 929 B for the independent vector bank. Native coordinate code is functionally equivalent at 1,243 B. The reported retrained-core comparator uses 926 B and preserves task outputs in this constructed case.

**C:** The new operand is simply a vector in the known native basis. The full cost of the shared basis and codes exceeds storing the explicit vector, and native coordinate encoding exactly accounts for the Mirror result.

**U:** Learned mechanisms, noisy support data, continual interference under actual updates, natural reasoning tasks, and fresh seeds. Fresh remained sealed.

## Evidence separation

- **Fact:** Byte sizes, hashes, output errors, retention scores and replay values are in `RESULTS_CORE.csv`.
- **Interpretation:** This controlled example demonstrates addressability but no useful storage gain for one new operand.
- **Hypothesis:** Shared content codes may amortize across many added operands; this result does not establish that scaling behavior.
