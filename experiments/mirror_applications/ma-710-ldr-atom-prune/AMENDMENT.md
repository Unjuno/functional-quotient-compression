# MA-710 amendment — learn the overcomplete atom bank

The first development implementation used a fixed random Toeplitz atom bank and therefore did not test the registered “learn-many atoms” proposal. Its source and development results are retained under `source/attempts/fixed_random_bank_not_learned/`; they are not evidence for MA-710.

Before any audit seed was opened, the implementation and protocol were amended to learn a 32-component Toeplitz-generator dictionary from support-fitted task generators using `sklearn.decomposition.DictionaryLearning` (LARS fit, OMP transform with two nonzero codes, 200 iterations). Audit settings are now fixed in `PROTOCOL.json`. This is an explicit development-stage amendment; fresh/audit tasks remain unopened until the amended source passes development checks.
