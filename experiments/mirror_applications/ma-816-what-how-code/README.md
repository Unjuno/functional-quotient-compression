# MA-816 — what × how computation coordinates

## H / T / D / C / U

**H:** Separate content (“what”) and mechanism (“how”) Views recover heldout cross-products with lower complete bytes and better function-per-byte than independent operators and native tensor factorization.

**T:** Synthetic 8D operator bank; six content values × five mechanisms; rank-2 separable teacher, 24 train and 6 heldout pairs; 1,536 heldout vectors; seed 81601; zero updates. Compared independent pair operators, shared core, native factorization and Mirror factors.

**D: FAIL.** Mirror and native factorization both have zero heldout error at 1,724 B and 1,728 B respectively, with same MAC proxy. The shared core alone reaches nRMSE .00947 at only 723 B, passing the .02 quality gate. Therefore the extra factor state does not improve the tested quality/storage frontier; Mirror also exactly aliases native tensor factorization.

**C:** The selected small-amplitude operator changes are already within the quality tolerance of the shared core. Where exact recovery is required, native factorization reproduces Mirror with slightly fewer bytes.

**U:** Learned reasoning systems, task-level utility, continual learning, and fresh seeds. Fresh remained sealed.

## Evidence separation

- **Fact:** Payload sizes, hashes, heldout error and replay checks are in `RESULTS_CORE.csv`.
- **Interpretation:** Under this tolerance, sharing alone is a better frontier point; exact factor codes add no Mirror-specific function.
- **Hypothesis:** Stronger interaction terms may require private capacity, but would need to beat native CP/Tucker controls at equal full bytes.
