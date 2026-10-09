# MA-487 — LISTA router for Mirror atom coefficients

Status: **FAIL**.

**H:** A dev-trained fixed-depth LISTA encoder on the same dictionary should approach OMP top-8 sparse reconstruction with lower inference depth while fitting a useful stored coordinate representation.

**T:** 32D synthetic functions from the shared 64-atom dictionary; LISTA trained only on worlds 48700-48701. Fresh worlds 48710-48712 × seeds 0-2 compared OMP top-8 with LISTA at 1/2/4/8 steps. Router weights, dictionary, and output indices/values are all serialized and charged.

**D:** FAIL. OMP mean NRMSE was .0269 at 15,329B. LISTA errors were .821/.742/.644/.358 for 1/2/4/8 steps, respectively; payload was 214,613B at every depth because router weights dominate. No LISTA setting approached the preregistered .05 quality threshold, and it used about 14× OMP bytes.

**C:** The tiny fixed development corpus and poorly conditioned learned unrolled inference network did not recover sparse codes; its dense step matrices are expensive. This implementation is a screening failure, not evidence against all LISTA variants.

**U:** tuning/init variants on development, convolutional or structured LISTA, learned dictionary, and non-synthetic tasks remain untested. Fresh results were not used for tuning.
