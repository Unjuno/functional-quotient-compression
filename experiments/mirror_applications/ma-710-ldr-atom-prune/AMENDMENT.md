# MA-710 amendment — support-learned atom bank and contaminated audit IDs

The first development implementation used a fixed random Toeplitz atom bank and therefore did not test the registered “learn-many atoms” proposal. It also accidentally ran its all-seed script, exposing audit IDs 71011–71013. Its source, results and audit payloads are retained under `source/attempts/fixed_random_bank_not_learned/`; they are invalid for the amended learned-bank method and the IDs are contaminated.

Before evaluating the amended method, the protocol was changed to learn a 32-component Toeplitz-generator dictionary from support-fitted task generators using `sklearn.decomposition.DictionaryLearning` (LARS fit, OMP transform with two nonzero codes). Development compared alpha 1e-8, .1, and 1.0. Alpha .1 was selected because it had the best post-fit-pruned development quality among those tested, though it still missed the frozen gates. The final value, iteration cap, tolerance and seed rule are in `PROTOCOL.json`.

The amended learned-bank method was evaluated only on development seed 71001. It missed quality/storage gates. No valid audit was run; the previously exposed audit IDs will not be reused. This is a development FAIL, not fresh evidence.
