# MA-810 — compositional motor option bank

## H / T / D / C / U

**H:** A shared rank-2 adapter basis with per-option Mirror coefficients preserves heldout option-sequence behavior using fewer complete bytes than a full adapter bank and improves on ordinary low-rank coefficient factorization.

**T:** Synthetic 8D linear motor core, eight options, rank-2 option residuals; ten training and eight heldout four-option sequences; seed 81001; 1,280 train and 2,048 test vectors; zero optimizer updates. Compared full independent adapters, shared core, native rank-2 basis coefficients, and Mirror option codes.

**D: FAIL for Mirror-specific attribution.** Full adapter bank uses 2,614 B. Mirror codes use 1,730 B (33.8% fewer) with heldout sequence nRMSE 0. The native rank-2 table uses 1,744 B with the same exact nRMSE, MAC proxy and decoded maps. Thus the result demonstrates low-rank adapter-bank compressibility in this constructed world, but the candidate is exactly the ordinary native low-rank coefficient representation. The Mirror code saves 14 B only through this NPZ field layout; both contain the same functional state.

**C:** Standard low-rank adapter factorization fully explains the apparent compression and compositional generalization. Field-name-dependent archive overhead is not a meaningful Mirror advantage.

**U:** Trained recurrent option policies, learned option discovery, physical rollouts, safety, and fresh seeds. Fresh remained sealed after exact native-function alias.

## Evidence separation

- **Fact:** Payload bytes, hashes, heldout sequence error, active MAC proxy, and exact payload reconstruction are in `RESULTS_CORE.csv`.
- **Interpretation:** Shared atoms preserve the selected synthetic sequence behavior while compressing the full bank, but standard low-rank adapter bases achieve the same function.
- **Hypothesis:** Any useful follow-up needs a Mirror coordinate that adds compositional behavior beyond native low-rank adapters, without relying on archive key names or synthetic oracle atoms.
