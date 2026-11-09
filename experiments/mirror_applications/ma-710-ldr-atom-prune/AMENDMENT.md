# MA-710 amendment — support-learned LDR atom bank

The first development implementation used a fixed random Toeplitz atom bank and therefore did not test the registered “learn-many atoms” proposal. Its source and results are retained under `source/attempts/fixed_random_bank_not_learned/`; they are not evidence for MA-710.

Before any audit seed was opened, the implementation and protocol were amended to learn a 32-component Toeplitz-generator dictionary from support-fitted task generators using `sklearn.decomposition.DictionaryLearning` (LARS fit, OMP transform with two nonzero codes). Development compared alpha 1e-8, .1, and 1.0. Alpha .1 was selected because it had the best development quality among those tested, though it still missed the frozen gates. The final value, 500 iteration cap, tolerance, and seed rule are frozen in `PROTOCOL.json`. This remains a development-only FAIL; audit seeds stay unopened unless the frozen development gate is met.
